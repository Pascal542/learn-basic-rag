from langchain_ollama import OllamaEmbeddings

class RetrievalEmbeddings(OllamaEmbeddings):
    """Añade los prefijos de recuperación de EmbeddingGemma antes de llamar a Ollama."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        formatted_texts = []

        for text in texts:
            formatted_texts.append(
                f"title: none | text: {text}"
            )

        return super().embed_documents(formatted_texts)


    def embed_query(self, text: str) -> list[float]:
        formatted_text = f"task: search result | query: {text}"

        return super().embed_query(formatted_text)