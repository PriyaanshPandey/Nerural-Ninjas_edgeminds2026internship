# Scholar Minds - Offline Research Paper RAG Engine
import argparse
import os

from rag.pdf_parser import load_documents
from rag.chunking import make_chunks
from rag.retriever import ChromaRetriever
from rag.generator import plan_questions, summarize_question, compile_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--model", default="llama3.2:1b")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--papers-dir", default="papers")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="output/report.md")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--reset-index", action="store_true")
    args = parser.parse_args()

    docs = load_documents(args.papers_dir)
    if not docs:
        raise SystemExit("No papers found. Put .pdf/.txt/.md files in ./papers")

    chunks = make_chunks(docs)
    retriever = ChromaRetriever(
        persist_dir=args.persist_dir,
        collection_name="papers",
        embed_model=args.embed_model,
    )
    if args.reset_index:
        retriever.reset()
    retriever.index_chunks(chunks)

    questions = plan_questions(args.topic, args.model, n=3)

    sections = []
    for q in questions:
        hits = retriever.query(q, top_k=args.top_k)
        ans = summarize_question(q, hits, args.model)
        sections.append({"question": q, "answer": ans})

    report = compile_report(args.topic, sections, args.model)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Research Report\n\n")
        f.write(report + "\n")

    print("Report written to:", args.out)


if __name__ == "__main__":
    main()