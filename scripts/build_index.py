import argparse
from pathlib import Path
from langchain_chroma import Chroma
from learn_basic_rag.embeddings import RetrievalEmbeddings
from learn_basic_rag.ingestion import load_documents, split_documents

def main():
    parser = argparse.ArgumentParser(description="Indexa los Markdown de data/documents.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Borra solo learning_documents antes de reconstruirla; no es incremental.",
    )
    args = parser.parse_args()
    base_path = Path(__file__).resolve().parent.parent
    documents_path = base_path / "data" / "documents"
    storage_path = base_path / ".chroma"

    if not documents_path.is_dir():
        raise FileNotFoundError("No existe la carpeta de documentos")

    documents = load_documents(documents_path)

    if not documents:
        raise ValueError("No se encontraron documentos Markdown")

    chunks = split_documents(documents)

    embeddings = RetrievalEmbeddings(
        model="embeddinggemma:latest",
        base_url="http://127.0.0.1:11434",
    )

    store = Chroma(
        collection_name="learning_documents",
        embedding_function=embeddings,
        persist_directory=str(storage_path),
    )

    # Usar --reset al cambiar o eliminar documentos evita registros obsoletos.
    if args.reset:
        store.reset_collection()

    ids = []
    for position, chunk in enumerate(chunks):
        source = chunk.metadata["source"]
        ids.append(f"{source}:{position}")

    store.add_documents(documents=chunks, ids=ids)


    print("Documentos cargados:", len(documents))
    print("Chunks enviados al índice:", len(chunks))
    print("Colección: learning_documents")

if __name__ == "__main__":
    main()