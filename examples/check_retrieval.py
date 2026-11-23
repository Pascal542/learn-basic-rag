from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from learn_basic_rag.embeddings import RetrievalEmbeddings

def main():
    embeddings = RetrievalEmbeddings(
        model = "embeddinggemma:latest",
        base_url = "http://127.0.0.1:11434"
    )

    storage_path = Path(__file__).resolve().parent.parent / ".chroma"

    store = Chroma(
        collection_name = "retrieval_smoke_test",
        embedding_function = embeddings,
        persist_directory = str(storage_path)
    )

    documents = [
        Document(
            page_content="La garantía del servicio dura treinta días.",
            metadata={"source": "garantias.md"},
        ),
        Document(
            page_content="Atendemos de lunes a viernes.",
            metadata={"source": "horarios.md"},
        ),
    ]

    store.add_documents(
        documents = documents,
        ids = ["demo-garantia", "demo-horario"]
    )

    results = store.similarity_search(
        "¿Cuánto dura la garantía?",
        k = 1
    )

    assert len(results) == 1

    for document in results:
        print("Texto recuperado:", document.page_content)
        print("Fuente:", document.metadata["source"])

if __name__ == "__main__":
    main()