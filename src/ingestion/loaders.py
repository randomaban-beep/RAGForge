from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


def load_document(file_path: str):
    """
    Load a document based on its file extension.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    print(
        f"Loading {path.name}..."
    )

    # -------------------------------
    # PDF
    # -------------------------------

    if extension == ".pdf":

        loader = PyPDFLoader(
            str(path)
        )

    # -------------------------------
    # DOCX
    # -------------------------------

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(path)
        )

    # -------------------------------
    # TXT / Markdown
    # -------------------------------

    else:

        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    documents = loader.load()

    # Add consistent metadata
    for document in documents:

        document.metadata["source"] = (
            path.name
        )

        document.metadata["file_type"] = (
            extension
        )

    print(
        f"Loaded {len(documents)} document/page(s)"
    )

    return documents