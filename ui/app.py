import logging
import streamlit as st
from learn_basic_rag.generation import generate_answer
from learn_basic_rag.retrieval import search_documents


logger = logging.getLogger(__name__)
st.set_page_config(
    page_title="Learn Basic RAG",
    page_icon="📚",
)

st.title("📚 Learn Basic RAG")
st.caption("LangChain · Chroma · Ollama")

st.info(
    "Demo educativa con documentos ficticios. "
    "Cada pregunta se procesa de forma independiente: "
    "el historial visible no se envía al modelo."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Limpiar conversación", key="clear_chat"):
    st.session_state.messages = []

st.sidebar.caption(
    "Limpiar la conversación no elimina los documentos ni el índice."
)


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Ver fragmentos recuperados"):
                st.caption(
                    "Estos son los candidatos usados como contexto. "
                    "No representan una verificación automática de las citas."
                )
                for source in message["sources"]:
                    st.markdown(f"**Fuente:** {source['name']}")
                    st.text(source["text"])


for message in st.session_state.messages:
    render_message(message)

question = st.chat_input(
    "Pregunta sobre los documentos...",
    max_chars=2000,
    key="question",
    submit_mode="disable",
)

if question and question.strip():

    question = question.strip()

    user_message = {
        "role": "user",
        "content": question,
    }

    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.spinner("Recuperando evidencia y generando respuesta..."):

            documents = search_documents(question, k=1)
            answer = generate_answer(question, documents)  
        sources = []
        
        for document in documents:
            sources.append(
                {
                    "name": document.metadata["source"],
                    "text": document.page_content,
                }
            )
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
        st.session_state.messages.append(assistant_message)
        render_message(assistant_message)
    except FileNotFoundError:
        st.error(
            "No se encontró el índice. Desde la raíz del proyecto, "
            "ejecuta: uv run python -m scripts.build_index"
        )
    except Exception:
        logger.exception("Error al ejecutar el RAG desde Streamlit")
        st.error(
            "No se pudo completar la consulta. Comprueba que Ollama "
            "esté funcionando y que exista la colección learning_documents. "
            "El detalle del error está en la terminal."
        )