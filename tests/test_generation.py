"""Pruebas del contrato de generación: no ejecutan Ollama."""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from learn_basic_rag import generation


def test_empty_evidence_does_not_call_model(monkeypatch):
    def unexpected_model(**kwargs):
        raise AssertionError("No debe llamarse al modelo sin documentos")

    monkeypatch.setattr(generation, "ChatOllama", unexpected_model)

    assert generation.generate_answer("¿Precio?", []) == (
        "No encuentro esa información en los documentos."
    )


def test_prompt_contains_question_evidence_and_source(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            captured["messages"] = messages
            return AIMessage(content="Respuesta simulada")

    monkeypatch.setattr(generation, "ChatOllama", FakeChat)
    document = Document(
        page_content="La garantía dura treinta días.",
        metadata={"source": "garantias.md"},
    )

    answer = generation.generate_answer("¿Cuánto dura la garantía?", [document])
    human_message = next(m for m in captured["messages"] if m.type == "human")

    assert "¿Cuánto dura la garantía?" in human_message.content
    assert document.page_content in human_message.content
    assert "garantias.md" in human_message.content
    assert answer == "Respuesta simulada"
