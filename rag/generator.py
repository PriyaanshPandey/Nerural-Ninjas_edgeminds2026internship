import re
import ollama


def _chat(model: str, prompt: str, system: str = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()


def plan_questions(topic: str, model: str, n: int = 3):
    system_prompt = (
        "You are a research planner. List focused sub-questions a researcher would ask about a topic. "
        "Output ONLY the questions, one per line, with no numbers, bullet points, or extra text."
    )
    prompt = f"List exactly {n} sub-questions about: {topic}"
    text = _chat(model, prompt, system=system_prompt)
    lines = [re.sub(r"^\d+[\.\)]\s*", "", l).strip() for l in text.splitlines()]
    questions = [l for l in lines if l]
    return questions[:n] if questions else [topic]


def deduplicate_lines(text: str) -> str:
    lines = text.split("\n")
    seen = set()
    unique_lines = []
    for line in lines:
        cleaned = line.strip()
        # Clean leading list markers recursively
        prev = ""
        while cleaned != prev:
            prev = cleaned
            cleaned = re.sub(r'^([-\*\+]\s+)', '', cleaned)
            cleaned = re.sub(r'^(\d+[\.\)]\s+)', '', cleaned)
        
        normalized = cleaned.strip().lower()
        if normalized:
            if normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
        else:
            unique_lines.append(line)
    return "\n".join(unique_lines)


def summarize_question(question: str, chunks, model: str):
    if not chunks:
        return "No relevant content found in local papers."
    
    # Sort chunks by source and index to preserve natural document reading order
    sorted_chunks = sorted(chunks, key=lambda c: (c.source, c.index))
    
    # Format context blocks cleanly without any raw '#chunk' markers
    context_parts = []
    for i, c in enumerate(sorted_chunks):
        context_parts.append(f"--- Passage {i+1} (Source File: {c.source}) ---\n{c.text}")
    context = "\n\n".join(context_parts)
    
    system_prompt = (
        "You are a precise academic assistant. You must answer relying ONLY on the provided context. "
        "Do NOT change variables (such as 'i' to 'n'). Copy formulas character-for-character from the text. "
        "When reading tables or text containing lists of numbers, be extremely careful. "
        "Do NOT confuse parameter count (e.g. 350M parameters), layer count (e.g. 24 layers), and token counts. "
        "Provide only the direct factual answer. Do NOT include any chatty text, introductory phrases, "
        "preambles, meta-commentary, or post-answer notes."
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a direct and concise answer based strictly on the context. Do NOT include any conversational filler, intro phrases, or post-answer notes. "
        "If a formula or equation is requested, extract and output all parts of the formula exactly as they are written in the context, without omitting, renaming, or altering anything (for example, keep the variable 'i', keep '2i' and '2i+1', and do not change '2i/dmodel' to '2n/dmodel' or similar). "
        "If the output contains multiple formulas, equations, or list items, format them beautifully. Write each item on its own separate line as a markdown bullet point (starting with a '-') so they are clearly separated and easy to read. Do NOT write them together on the same line. "
        "Cite the source file names in brackets like [filename.pdf] when referencing information, "
        "but do NOT include any chunk numbers, paragraph indices, or passage markers (like #chunk) in your response."
    )
    ans = _chat(model, prompt, system=system_prompt)
    return deduplicate_lines(ans)


def compile_report(topic: str, sections, model: str):
    system_prompt = (
        "You are an academic research compiler. Write a structured, concise research report. "
        "Start with a short summary, then include the sections below."
    )
    joined = "\n\n".join([f"## {s['question']}\n{s['answer']}" for s in sections])
    prompt = (
        f"Topic: {topic}\n\n"
        f"Sections:\n{joined}"
    )
    return _chat(model, prompt, system=system_prompt)
