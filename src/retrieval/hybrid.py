from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rank_bm25 import BM25Okapi


# =========================================================
# DEFAULT DOCUMENT
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "sample.txt"


# =========================================================
# DOCUMENT LOADING
# =========================================================

def load_document(file_path):
    """
    Load a single document based on its file extension.

    Supported:
        .txt
        .md
        .pdf
        .docx
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    if extension in [".txt", ".md"]:

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    elif extension == ".pdf":

        loader = PyPDFLoader(
            str(file_path)
        )

    elif extension == ".docx":

        loader = Docx2txtLoader(
            str(file_path)
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader.load()


def load_documents(file_paths=None):
    """
    Load documents.

    If file_paths is None:
        load data/sample.txt

    Otherwise:
        load all supplied files.
    """

    if file_paths is None:

        return load_document(DATA_PATH)

    documents = []

    for file_path in file_paths:

        documents.extend(
            load_document(file_path)
        )

    return documents


# =========================================================
# CHUNKING
# =========================================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    return splitter.split_documents(
        documents
    )


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embeddings():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# VECTOR STORE
# =========================================================

def create_vector_store(
    chunks,
    embeddings=None,
):

    if embeddings is None:

        embeddings = create_embeddings()

    # -----------------------------------------------------
    # IMPORTANT:
    # Use in-memory Qdrant.
    #
    # This avoids the:
    #
    # "storage folder is already accessed by another
    # instance of Qdrant client"
    #
    # error.
    # -----------------------------------------------------

    client = QdrantClient(
        ":memory:"
    )

    vector_size = len(
        embeddings.embed_query(
            "test query"
        )
    )

    collection_name = "ragforge"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    if chunks:

        vector_store.add_documents(
            chunks
        )

    return vector_store


# =========================================================
# BM25
# =========================================================

def create_bm25(chunks):

    tokenized_documents = [

        document.page_content
        .lower()
        .split()

        for document in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_search(
    query,
    chunks,
    vector_store,
    bm25,
    k=4,
    rrf_k=60,
):
    """
    Hybrid retrieval using Reciprocal Rank Fusion.

    Combines:

        Dense retrieval
        +
        BM25 keyword retrieval
    """

    # =====================================================
    # DENSE RETRIEVAL
    # =====================================================

    dense_results = vector_store.similarity_search(
        query,
        k=min(k, len(chunks)),
    )

    # =====================================================
    # BM25 RETRIEVAL
    # =====================================================

    tokenized_query = (
        query.lower().split()
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:k]

    bm25_results = [

        chunks[i]

        for i in bm25_indices
    ]

    # =====================================================
    # RECIPROCAL RANK FUSION
    # =====================================================

    scores = {}

    documents = {}

    # -----------------------------------------------------
    # Dense ranking
    # -----------------------------------------------------

    for rank, document in enumerate(
        dense_results,
        start=1,
    ):

        key = document.page_content

        scores[key] = (
            scores.get(key, 0)
            + 1 / (rrf_k + rank)
        )

        documents[key] = document

    # -----------------------------------------------------
    # BM25 ranking
    # -----------------------------------------------------

    for rank, document in enumerate(
        bm25_results,
        start=1,
    ):

        key = document.page_content

        scores[key] = (
            scores.get(key, 0)
            + 1 / (rrf_k + rank)
        )

        documents[key] = document

    # =====================================================
    # FINAL RANKING
    # =====================================================

    ranked_documents = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [

        documents[key]

        for key, score
        in ranked_documents
    ]


# =========================================================
# TEST
# =========================================================

def main():

    print(
        "Loading documents..."
    )

    documents = load_documents()

    print(
        f"Loaded {len(documents)} document(s)."
    )

    chunks = create_chunks(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    embeddings = create_embeddings()

    vector_store = create_vector_store(
        chunks,
        embeddings,
    )

    bm25 = create_bm25(
        chunks
    )

    query = (
        "What is hybrid retrieval?"
    )

    results = hybrid_search(
        query=query,
        chunks=chunks,
        vector_store=vector_store,
        bm25=bm25,
        k=4,
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        "\nHYBRID RETRIEVAL RESULTS:"
    )

    for i, document in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            document.page_content
        )


if __name__ == "__main__":
    main()