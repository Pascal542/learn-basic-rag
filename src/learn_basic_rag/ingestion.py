from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory: Path) -> list[Document]:
    """Lee Markdown de una carpeta y conserva el nombre como fuente."""
    documents = []

    for file_path in sorted(directory.glob("*.md")):
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8")

        document = Document(
            page_content=text,
            metadata={"source": file_path.name}
        )

        documents.append(document)

    return documents


def split_documents(
        documents: list[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 80
    ) -> list[Document]:
    """Fragmenta por caracteres, con solapamiento y metadatos de origen."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    return splitter.split_documents(documents)
