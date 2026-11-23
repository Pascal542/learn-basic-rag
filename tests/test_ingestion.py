from learn_basic_rag.ingestion import load_documents, split_documents
from langchain_core.documents import Document


def test_load_documents_reads_markdown_and_keeps_source(tmp_path):
    # Datos ficticios de prueba, no políticas reales.
    text = "La garantía del servicio es de 30 días."

    document_path = tmp_path / "garantias.md"
    document_path.write_text(text, encoding="utf-8")

    ignored_path = tmp_path / "notas.txt"
    ignored_path.write_text("Ignorar este archivo.", encoding="utf-8")


    documents = load_documents(tmp_path)

    assert len(documents) == 1
    document = next(iter(documents))

    assert document.page_content == text
    assert document.metadata["source"] == "garantias.md"


def test_split_documents_limits_size_and_keeps_source():
    """Test Text"""
    text = "La garantia cubre defectos del servicio. " * 20

    document = Document(
        page_content = text,
        metadata = {"source": "garantias.md"}
    )

    chunks = split_documents(
        documents = [document],
        chunk_size = 100,
        chunk_overlap = 20 
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.page_content.strip()
        assert len(chunk.page_content) <= 100
        assert chunk.metadata["source"] == "garantias.md"


