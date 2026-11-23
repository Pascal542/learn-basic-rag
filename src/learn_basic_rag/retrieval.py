from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from learn_basic_rag.embeddings import RetrievalEmbeddings


def search_documents(query: str, k: int = 1) -> list[Document]:
    """Devuelve los k candidatos más cercanos; no garantiza evidencia suficiente."""
    base_path = Path(__file__).resolve().parent.parent.parent
    storage_path = base_path / ".chroma"

    if not storage_path.is_dir():
        raise FileNotFoundError("Primero debes ejecutar build_index.py")

    embeddings = RetrievalEmbeddings(
        model="embeddinggemma:latest",
        base_url="http://127.0.0.1:11434",
    )

    store = Chroma(
        collection_name="learning_documents",
        embedding_function=embeddings,
        persist_directory=str(storage_path),
        create_collection_if_not_exists=False,
    )

    return store.similarity_search(query, k=k)