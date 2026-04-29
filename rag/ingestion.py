from __future__ import annotations

import os
from typing import List, Optional

import chromadb
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def sanitize_collection_name(name: str) -> str:
    """ChromaDB 1.5+ n'accepte que [a-zA-Z0-9._-], min 3 chars."""
    import re
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name.strip())
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = sanitized + "_col"
    return sanitized[:512]

_embeddings: Optional[HuggingFaceEmbeddings] = None
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client


def _split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def _embed_and_store(chunks: List[Document], collection_name: str) -> None:
    client = _get_client()
    embeddings = _get_embeddings()
    safe_name = sanitize_collection_name(collection_name)
    collection = client.get_or_create_collection(safe_name)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"{safe_name}_{i}_{abs(hash(t))}" for i, t in enumerate(texts)]
    vectors = embeddings.embed_documents(texts)

    collection.add(documents=texts, embeddings=vectors, metadatas=metadatas, ids=ids)


def ingest_pdf(file_path: str, collection_name: str) -> int:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    source_name = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["collection"] = collection_name
    chunks = _split(docs)
    _embed_and_store(chunks, collection_name)
    return len(chunks)


def ingest_url(url: str, collection_name: str) -> int:
    loader = WebBaseLoader(url)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = url
        doc.metadata["collection"] = collection_name
        doc.metadata.pop("page", None)
    chunks = _split(docs)
    _embed_and_store(chunks, collection_name)
    return len(chunks)


def get_all_docs(collection_name: str) -> List[Document]:
    client = _get_client()
    safe_name = sanitize_collection_name(collection_name)
    try:
        collection = client.get_collection(safe_name)
    except Exception:
        return []
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


def list_sources(collection_name: str) -> List[str]:
    docs = get_all_docs(collection_name)
    return sorted({d.metadata.get("source", "inconnu") for d in docs})


def delete_source(source_name: str, collection_name: str) -> None:
    client = _get_client()
    safe_name = sanitize_collection_name(collection_name)
    try:
        collection = client.get_collection(safe_name)
    except Exception:
        return
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        results["ids"][i]
        for i, meta in enumerate(results["metadatas"])
        if meta.get("source") == source_name
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)


def list_collections() -> List[str]:
    client = _get_client()
    return [c.name for c in client.list_collections()]


def delete_collection(collection_name: str) -> None:
    client = _get_client()
    safe_name = sanitize_collection_name(collection_name)
    try:
        client.delete_collection(safe_name)
    except Exception:
        pass
