import os
import tempfile

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from rag.ingestion import (
    delete_collection,
    delete_source,
    ingest_pdf,
    ingest_url,
    list_collections,
    list_sources,
)
from rag.retriever import build_retriever
from rag.graph import build_graph

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = []


def get_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key, streaming=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Chatbot")

    # Clé API
    st.subheader("🔑 OpenAI API Key")
    api_key = st.text_input("Clé API", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    # Gestion des collections
    st.subheader("📂 Collection de documents")
    existing_collections = list_collections()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        new_col_name = st.text_input("Nouvelle collection", placeholder="ex: Projet A")
    with col_b:
        st.write("")
        st.write("")
        if st.button("Créer", use_container_width=True):
            if new_col_name.strip():
                st.session_state.collection = new_col_name.strip()
                reset_chat()
                st.rerun()

    if existing_collections:
        selected = st.selectbox(
            "Collection active",
            options=existing_collections,
            index=(
                existing_collections.index(st.session_state.collection)
                if st.session_state.collection in existing_collections
                else 0
            ),
        )
        if selected != st.session_state.collection:
            st.session_state.collection = selected
            reset_chat()
            st.rerun()

    collection = st.session_state.collection

    st.divider()

    if collection:
        # Upload PDF
        st.subheader("📄 Ajouter des PDFs")
        uploaded_files = st.file_uploader(
            "Glisser-déposer vos PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                existing = list_sources(collection)
                if uploaded_file.name not in existing:
                    with st.spinner(f"Ingestion de {uploaded_file.name}..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name
                        try:
                            n = ingest_pdf(tmp_path, collection)
                            st.success(f"{uploaded_file.name} — {n} chunks indexés")
                        finally:
                            os.unlink(tmp_path)
                else:
                    st.info(f"{uploaded_file.name} déjà indexé.")

        st.divider()

        # Ajout URL
        st.subheader("🌐 Ajouter une URL")
        url_input = st.text_input("URL", placeholder="https://...")
        if st.button("Indexer l'URL", use_container_width=True):
            if url_input.strip():
                with st.spinner(f"Ingestion de {url_input}..."):
                    try:
                        n = ingest_url(url_input.strip(), collection)
                        st.success(f"URL indexée — {n} chunks")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

        st.divider()

        # Documents indexés
        st.subheader("📚 Documents indexés")
        sources = list_sources(collection)
        if sources:
            for src in sources:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"• `{src}`")
                if col2.button("🗑", key=f"del_{src}"):
                    delete_source(src, collection)
                    st.rerun()

            st.divider()
            if st.button("🗑 Vider la collection", use_container_width=True, type="secondary"):
                delete_collection(collection)
                st.session_state.collection = None
                reset_chat()
                st.rerun()
        else:
            st.caption("Aucun document indexé dans cette collection.")
    else:
        st.info("Créez ou sélectionnez une collection pour commencer.")


# ── Main — Chat ───────────────────────────────────────────────────────────────
st.title("💬 Chatbot RAG")

if not api_key:
    st.warning("Entrez votre clé OpenAI dans la barre latérale pour commencer.")
    st.stop()

if not collection:
    st.info("Créez une collection dans la barre latérale et ajoutez des documents.")
    st.stop()

sources_check = list_sources(collection)
if not sources_check:
    st.info("Ajoutez des PDFs ou des URLs dans la barre latérale pour commencer.")
    st.stop()

# Affichage de l'historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources utilisées"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Bouton reset
if st.session_state.messages:
    if st.button("🔄 Nouvelle conversation", use_container_width=False):
        reset_chat()
        st.rerun()

# Input utilisateur
if question := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Recherche en cours..._")

        llm = get_llm(api_key)
        retriever = build_retriever(collection)

        if retriever is None:
            answer = "Aucun document trouvé dans cette collection."
            sources = []
        else:
            graph = build_graph(llm, retriever)
            result = graph.invoke(
                {
                    "question": question,
                    "rewritten_question": "",
                    "chat_history": st.session_state.chat_history,
                    "retrieved_docs": [],
                    "grade": "",
                    "retry_count": 0,
                    "answer": "",
                    "sources": [],
                    "collection": collection,
                }
            )
            answer = result["answer"]
            sources = result.get("sources", [])

        placeholder.markdown(answer)
        if sources:
            with st.expander("📎 Sources utilisées"):
                for src in sources:
                    st.markdown(f"- {src}")

    # Mise à jour de l'historique
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
