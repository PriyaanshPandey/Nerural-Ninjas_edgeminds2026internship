import re
from dataclasses import dataclass


@dataclass
class Chunk:
    source: str
    index: int
    text: str
    page: int = 1


def chunk_text(text: str, size: int = 1500, overlap: int = 500):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def make_chunks(docs):
    all_chunks = []
    from collections import defaultdict
    docs_by_source = defaultdict(list)
    for doc in docs:
        docs_by_source[doc["source"]].append(doc)
        
    for source, source_docs in docs_by_source.items():
        source_docs.sort(key=lambda d: d.get("page", 1))
        chunk_idx = 0
        for doc in source_docs:
            parts = chunk_text(doc["text"])
            for part in parts:
                all_chunks.append(
                    Chunk(
                        source=source,
                        index=chunk_idx,
                        page=doc.get("page", 1),
                        text=part,
                    )
                )
                chunk_idx += 1
    return all_chunks