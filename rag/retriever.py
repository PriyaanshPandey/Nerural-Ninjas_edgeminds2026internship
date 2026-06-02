import faiss
import numpy as np


def build_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

def retrieve(query, chunks, index, model, top_k=5):

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return results

from sentence_transformers import SentenceTransformer
from pdf_parser import extract_text
from chunking import create_chunks
from embeddings import create_embeddings

if __name__ == "__main__":

    pages = extract_text(
        "data/sample_papers/sample_paper.pdf"
    )

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    index = build_index(embeddings)

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    query = "What is the Transformer architecture?"

    results = retrieve(
        query,
        chunks,
        index,
        model
    )

    print("\nQuery:", query)

    for i, chunk in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Page:", chunk["page"])
        print(chunk["text"][:500])