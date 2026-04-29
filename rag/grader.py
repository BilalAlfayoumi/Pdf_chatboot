import json
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

GRADER_PROMPT = """Tu es un évaluateur de pertinence documentaire. Sois TRÈS PERMISSIF.
Ta tâche : déterminer si les documents contiennent AU MOINS QUELQUES informations liées à la question.

Règle : réponds "non pertinent" UNIQUEMENT si les documents sont totalement hors-sujet (ex: question sur la cuisine, documents sur la physique).
Dans tous les autres cas, réponds "pertinent" — même pour des questions générales comme "de quoi parle ce document ?", "résume-moi ça", "explique-moi le concept X".

Question : {question}

Documents (extraits) :
{documents}

Réponds UNIQUEMENT en JSON, sans aucun texte autour :
{{"score": "pertinent"}} ou {{"score": "non pertinent"}}"""


def grade_documents(question: str, docs: List[Document], llm: BaseChatModel) -> str:
    if not docs:
        return "non pertinent"

    docs_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content[:300]}"
        for d in docs
    )
    prompt = GRADER_PROMPT.format(question=question, documents=docs_text)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        # Extraire le JSON même s'il y a du texte autour
        import re
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            score = parsed.get("score", "pertinent")
            return score if score in ("pertinent", "non pertinent") else "pertinent"
        return "pertinent"
    except Exception:
        return "pertinent"
