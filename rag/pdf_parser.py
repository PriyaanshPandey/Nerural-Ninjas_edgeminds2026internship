import os
import re

try:
    import fitz  # PyMuPDF
except Exception as exc:
    raise SystemExit("PyMuPDF missing. Install: pip install pymupdf") from exc


def read_pdf(path: str):
    doc = fitz.open(path)
    pages = []
    for page_idx, page in enumerate(doc):
        blocks_text = []
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # Skip image blocks
                continue
            
            block_lines = []
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                
                # Find typical/standard size of the line
                longest_span = max(spans, key=lambda s: len(s["text"]))
                std_size = longest_span["size"]
                
                # Classify spans
                classified_spans = []
                for span in spans:
                    t = span["text"]
                    size = span["size"]
                    origin_x, origin_y = span["origin"]
                    
                    span_type = "normal"
                    if size < std_size - 1.0:
                        std_spans = [s for s in spans if abs(s["size"] - std_size) < 1.0]
                        if std_spans:
                            std_y = std_spans[0]["origin"][1]
                            if origin_y < std_y - 1.5:
                                span_type = "super"
                            elif origin_y > std_y + 1.0:
                                span_type = "sub"
                        else:
                            avg_y = sum(s["origin"][1] for s in spans) / len(spans)
                            if origin_y < avg_y - 1.0:
                                span_type = "super"
                            elif origin_y > avg_y + 1.0:
                                span_type = "sub"
                    
                    classified_spans.append({
                        "text": t,
                        "type": span_type
                    })
                
                # Group consecutive spans
                grouped_spans = []
                current_group = []
                current_type = None
                
                for item in classified_spans:
                    if current_type is None:
                        current_type = item["type"]
                        current_group.append(item["text"])
                    elif item["type"] == current_type:
                        current_group.append(item["text"])
                    else:
                        grouped_spans.append({
                            "text": "".join(current_group),
                            "type": current_type
                        })
                        current_type = item["type"]
                        current_group = [item["text"]]
                
                if current_group:
                    grouped_spans.append({
                        "text": "".join(current_group),
                        "type": current_type
                    })
                
                # Format spans
                line_text = ""
                for item in grouped_spans:
                    t = item["text"]
                    span_type = item["type"]
                    
                    t_clean = t.strip()
                    if not t_clean:
                        line_text += t
                        continue
                    
                    leading_space = " " if t.startswith(" ") else ""
                    trailing_space = " " if t.endswith(" ") else ""
                    
                    if span_type == "super":
                        formatted = f"{leading_space}^{{{t_clean}}}{trailing_space}"
                    elif span_type == "sub":
                        formatted = f"{leading_space}_{{{t_clean}}}{trailing_space}"
                    else:
                        formatted = t
                    
                    line_text += formatted
                    
                block_lines.append(line_text)
            
            blocks_text.append("\n".join(block_lines))
            
        page_text = "\n\n".join(blocks_text)
        
        patterns = [
            r"\n\s*References\s*\n",
            r"\n\s*Bibliography\s*\n"
        ]
        has_refs = False
        for pattern in patterns:
            if re.search(pattern, page_text, flags=re.IGNORECASE):
                page_text = re.split(pattern, page_text, flags=re.IGNORECASE)[0]
                has_refs = True
        
        if page_text.strip():
            pages.append({"text": page_text, "page": page_idx + 1})
            
        if has_refs:
            break
            
    return pages


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_documents(papers_dir: str):
    docs = []
    for root, _, files in os.walk(papers_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            if ext == ".pdf":
                pages = read_pdf(path)
                for p in pages:
                    docs.append({"source": name, "text": p["text"], "page": p["page"]})
            elif ext in [".txt", ".md"]:
                content = read_text(path)
                if content and content.strip():
                    docs.append({"source": name, "text": content, "page": 1})
    return docs