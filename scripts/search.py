from learn_basic_rag.retrieval import search_documents

def main():
    query = input("Pregunta: ").strip()

    if not query:
        raise ValueError("La pregunta no puede estar vacía")

    documents = search_documents(query)

    for document in documents:
        print("\nFuente:", document.metadata["source"])
        print("Texto recuperado:")
        print(document.page_content)


if __name__ == "__main__":
    main()