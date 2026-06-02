import fitz

def extract_text(pdf_path):
    pages = []

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]

        pages.append({
            "page": page_num + 1,
            "text": page.get_text()
        })

    return pages


if __name__ == "__main__":
    pdf_path = "data/sample_papers/sample_paper.pdf"

    pages = extract_text(pdf_path)

    print("Total Pages:", len(pages))
    print("\nFirst Page Preview:\n")
    print(pages[0]["text"][:1000])