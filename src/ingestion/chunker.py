from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


def chunk_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        documents
    )

    # Give every chunk an ID
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

    print(
        f"Created {len(chunks)} chunks"
    )

    return chunks