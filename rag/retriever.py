from __future__ import annotations

from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from rag.ingestion import CHROMA_PATH, get_all_docs, _get_embeddings, sanitize_collection_name

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 6
TOP_K_RERANKED = 4

_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def build_retriever(collection_name: str) -> Optional[BaseRetriever]:
    all_docs = get_all_docs(collection_name)
    if not all_docs:
        return None

    safe_name = sanitize_collection_name(collection_name)
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        collection_name=safe_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = TOP_K

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    return ensemble


def rerank_docs(query: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return docs
    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:TOP_K_RERANKED]]
