import ollama

from sentence_transformers import SentenceTransformer

from pdf_parser import extract_text
from chunking import create_chunks
from embeddings import create_embeddings
from retriever import build_index, retrieve


def generate_answer(question, retrieved_chunks):

    context = ""

    for chunk in retrieved_chunks:
        context += (
            f"\n[Page {chunk['page']}]\n"
            f"{chunk['text']}\n"
        )

    prompt = f"""
You are a research paper assistant.

Answer ONLY using the provided context.

Rules:
- Give a direct answer.
- Quote numbers exactly as they appear.
- If a number is requested, include the explanation around it.
- Do NOT use outside knowledge.
- If the answer is not present in the context, reply exactly:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


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

    question = (
        "What is the Transformer architecture?"
    )

    retrieved_chunks = retrieve(
        question,
        chunks,
        index,
        model,
        top_k=5
    )

    print("\nRETRIEVED CHUNKS:\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"\nResult {i}")
        print(f"Page: {chunk['page']}")
        print(chunk["text"][:300])
        print("-" * 50)

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    source_pages = sorted(
        list(
            set(chunk["page"] for chunk in retrieved_chunks)
        )
    )

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    print(source_pages)