import re
from src.models import Document, stable_id


def chunk_documents(documents, chunk_words=180, overlap_words=30):
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("Require 0 <= overlap_words < chunk_words.")
    chunks = []
    for doc in documents:
        words = list(re.finditer(r"\S+", doc.text))
        for start in range(0, len(words), chunk_words - overlap_words):
            end = min(start + chunk_words, len(words))
            text = doc.text[words[start].start() : words[end - 1].end()]
            chunks.append(
                Document(
                    stable_id(doc.id, str(start), text),
                    text,
                    {**doc.metadata, "parent_id": doc.id, "word_start": start, "word_end": end},
                )
            )
            if end == len(words):
                break
    return chunks
