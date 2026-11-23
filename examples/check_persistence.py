from pathlib import Path
from langchain_chroma import Chroma
from learn_basic_rag.embeddings import RetrievalEmbeddings



def main():
    storage_path = Path(__file__).resolve().parent.parent / ".chroma"

    if not storage_path.is_dir():
        raise FileNotFoundError("No existe el directorio del índice")

    embeddings = RetrievalEmbeddings(
        model="embeddinggemma:latest",
        base_url="http://127.0.0.1:11434",
    )

    store = Chroma(
        collection_name="retrieval_smoke_test",
        embedding_function=embeddings,
        persist_directory=str(storage_path),
        create_collection_if_not_exists=False,
    )

    results = store.similarity_search(
        "¿Qué días atienden?",
        k=1,
    )

    assert len(results) == 1

    document = next(iter(results))

    assert document.metadata["source"] == "horarios.md"
    assert document.page_content == "Atendemos de lunes a viernes."

    print("Texto recuperado:", document.page_content)
    print("Fuente:", document.metadata["source"])
    print("Persistencia y recuperación del caso de prueba: correctas")


if __name__ == "__main__":
    main()