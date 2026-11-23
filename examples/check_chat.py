from langchain_ollama import ChatOllama

def main():
    llm = ChatOllama(
        model = "qwen3:4b",
        base_url = "http://127.0.0.1:11434",
        temperature = 0,
        num_ctx = 4096
    )

    response = llm.invoke(
        [
            (
                "system",
                "Responde en español y de forma breve."
            ),
            (
                "human",
                "Responde únicamente con estas palabras: conexión correcta"
            )
        ]
    )

    print(response.content)

if __name__ == "__main__":
    main()