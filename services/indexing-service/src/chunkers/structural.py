import re


def structural_chunk(text: str) -> list[str]:
    sections = re.split(r"(?m)^#{1,3}\s+", text)
    headers = re.findall(r"(?m)^(#{1,3}\s+.+)", text)

    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        if i > 0 and i - 1 < len(headers):
            section = headers[i - 1].strip() + "\n\n" + section
        chunks.append(section)
    return [c for c in chunks if len(c.split()) > 10]
