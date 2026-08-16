import re


def semantic_chunk(text: str, max_chunk_size: int = 500) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para.split())
        if current and current_len + para_len > max_chunk_size:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))
    return chunks
