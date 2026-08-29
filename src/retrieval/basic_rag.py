from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


DATA_PATH = Path("data/sample.txt")


def load_documents():
    loader = TextLoader(str(DATA_PATH), encoding="utf-8")
    return loader.load()


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = QdrantClient(":memory:")

    vector_size = len(
        embeddings.embed_query("test query")
    )

    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name="ragforge",
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="ragforge",
        embedding=embeddings
    )

    vector_store.add_documents(chunks)

    return vector_store


def main():
    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} document(s)")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks")

    vector_store = create_vector_store(chunks)

    query = "What is hybrid retrieval?"

    results = vector_store.similarity_search(query, k=3)

    print("\nQUERY:")
    print(query)

    print("\nRETRIEVED CONTEXT:")

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content)


if __name__ == "__main__":
    main()