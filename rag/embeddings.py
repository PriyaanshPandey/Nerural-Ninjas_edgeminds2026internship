from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings

from pdf_parser import extract_text
from chunking import create_chunks

if __name__ == "__main__":

    pages = extract_text(
        "data/sample_papers/sample_paper.pdf"
    )

    chunks = create_chunks(pages)

    embeddings = create_embeddings(chunks)

    print("Number of embeddings:", len(embeddings))
    print("Embedding dimension:", len(embeddings[0]))