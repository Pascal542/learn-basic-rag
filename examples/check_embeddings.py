from langchain_ollama import OllamaEmbeddings

def main():
    embeddings = OllamaEmbeddings(
        model = "embeddinggemma:latest",
        base_url = "http://127.0.0.1:11434"
    )

    document_vectors = embeddings.embed_documents(
        [
            "title: none | text: La garantia del servicio dura treinta dias",
            "title: none | text: Atendemos de lunes a viernes"
        ]
    )

    query_vector = embeddings.embed_query(
        "task: search result | query: Cuanto dura la garantia?"
    )

    assert len(document_vectors) == 2
    assert query_vector

    for vector in document_vectors:
        assert len(vector) == len(query_vector)

    print("Vectores de documentos:", len(document_vectors))
    print("Dimensiones de la consulta:", len(query_vector))
    print("Comprobación de estructura: correcta")


if __name__ == "__main__":
    main()