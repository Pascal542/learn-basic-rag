from learn_basic_rag.generation import generate_answer
from learn_basic_rag.retrieval import search_documents


def main():
    question = input("Pregunta: ").strip()

    if not question:
        raise ValueError("La pregunta no puede estar vacía")

    documents = search_documents(question, k=1)

    answer = generate_answer(question, documents)

    print("\nRespuesta:")
    print(answer)


if __name__ == "__main__":
    main()
