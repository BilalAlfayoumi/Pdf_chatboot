from typing import List, Literal
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import END, START, StateGraph

from rag.grader import grade_documents
from rag.retriever import rerank_docs

MAX_RETRIES = 2

REWRITE_PROMPT = """Tu es un expert en reformulation de questions.
Étant donné l'historique de conversation et la question de l'utilisateur, reformule la question pour qu'elle soit autonome et compréhensible sans l'historique.
Si c'est une première question sans ambiguïté, retourne-la telle quelle.

Historique :
{chat_history}

Question originale : {question}

Question reformulée (retourne UNIQUEMENT la question, sans explication) :"""

REWRITE_RETRY_PROMPT = """La question précédente n'a pas permis de trouver des documents pertinents.
Reformule-la de façon plus générale ou avec des synonymes pour élargir la recherche.

Question précédente : {question}

Nouvelle formulation :"""

GENERATE_PROMPT = """Tu es un professeur bienveillant et pédagogue. Tu expliques les concepts avec des mots simples, des exemples concrets, et tu adaptes ton langage à l'étudiant.
Tu t'appuies sur les extraits de documents fournis pour répondre, mais tu parles de façon naturelle — comme si tu expliquais à voix haute, pas comme si tu lisais un rapport.

Règles importantes :
- Si la question est dans le contenu des documents : explique clairement, donne des exemples si possible, propose d'approfondir.
- Si la question est floue ou générale : reformule-la toi-même et réponds quand même du mieux possible avec ce que tu as.
- Si le sujet n'est vraiment pas du tout dans les documents : dis-le simplement et avec bienveillance, propose ce que tu peux faire à la place.
- Utilise le tutoiement, sois encourageant, et invite l'étudiant à poser des questions de suivi.

Extraits de documents :
{context}

{chat_history_section}

Question de l'étudiant : {question}

Réponse du professeur :"""

FALLBACK_MESSAGE = (
    "Hmm, je n'ai pas trouvé d'éléments précis dans les documents pour répondre à ça. "
    "Tu peux reformuler ta question différemment, ou vérifier que le bon document est bien chargé. "
    "N'hésite pas à me demander d'expliquer un autre point du cours !"
)


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    chat_history: List[BaseMessage]
    retrieved_docs: List[Document]
    grade: str
    retry_count: int
    answer: str
    sources: List[str]
    collection: str


def _format_chat_history(history: List[BaseMessage]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role = "Utilisateur" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role} : {msg.content}")
    return "\n".join(lines)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}"
        + (f", page {d.metadata['page'] + 1}" if "page" in d.metadata else "")
        + f"]\n{d.page_content}"
        for d in docs
    )


def _extract_sources(docs: List[Document]) -> List[str]:
    seen = set()
    sources = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page")
        label = f"{src}, page {page + 1}" if page is not None else src
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def build_graph(llm: BaseChatModel, retriever: BaseRetriever) -> StateGraph:

    def rewrite_query(state: RAGState) -> dict:
        history_str = _format_chat_history(state.get("chat_history", []))
        retry_count = state.get("retry_count", 0)

        if retry_count > 0:
            prompt = REWRITE_RETRY_PROMPT.format(
                question=state.get("rewritten_question") or state["question"]
            )
        else:
            prompt = REWRITE_PROMPT.format(
                chat_history=history_str or "Aucun",
                question=state["question"],
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"rewritten_question": response.content.strip()}

    def retrieve(state: RAGState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        raw_docs = retriever.invoke(query)
        reranked = rerank_docs(query, raw_docs)
        return {"retrieved_docs": reranked}

    def grade_docs_node(state: RAGState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        grade = grade_documents(query, state["retrieved_docs"], llm)
        retry_count = state.get("retry_count", 0)
        return {"grade": grade, "retry_count": retry_count + 1}

    def generate(state: RAGState) -> dict:
        docs = state.get("retrieved_docs", [])
        grade = state.get("grade", "pertinent")

        if grade == "non pertinent" and state.get("retry_count", 0) >= MAX_RETRIES:
            return {"answer": FALLBACK_MESSAGE, "sources": []}

        context = _format_docs(docs)
        history = _format_chat_history(state.get("chat_history", []))
        chat_section = f"Historique de conversation :\n{history}\n" if history else ""

        prompt = GENERATE_PROMPT.format(
            context=context,
            chat_history_section=chat_section,
            question=state["question"],
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        return {
            "answer": response.content,
            "sources": _extract_sources(docs),
        }

    def route_after_grade(state: RAGState) -> Literal["generate", "rewrite_query"]:
        if state["grade"] == "pertinent":
            return "generate"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "generate"
        return "rewrite_query"

    graph = StateGraph(RAGState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("generate", generate)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("generate", END)

    return graph.compile()
