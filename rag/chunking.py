def create_chunks(pages, chunk_size=200, overlap=50):
    chunks = []

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])

            chunks.append({
                "text": chunk_text,
                "page": page["page"]
            })

    return chunks



from pdf_parser import extract_text

if __name__ == "__main__":
    pages = extract_text(
        "data/sample_papers/sample_paper.pdf"
    )

    chunks = create_chunks(pages)

    print("Total Chunks:", len(chunks))

    print("\nFirst Chunk:\n")
    print(chunks[0]["text"])

    print("\nChunk Metadata:")
    print(chunks[0]["page"])