from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from reranker import Reranker


DATA_PATH = Path("data/sample.txt")


def main():

    loader = TextLoader(
        str(DATA_PATH),
        encoding="utf-8"
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    query = "What is hybrid retrieval?"

    reranker = Reranker()

    results = reranker.rerank(
        query,
        chunks,
        top_k=3
    )

    print("\nRERANKED RESULTS:")

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Rank {rank} | Score: {score:.4f} ---")
        print(document.page_content)


if __name__ == "__main__":
    main()