"""Three ways to cut a document into chunks."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_no: int
    heading: str
    text: str


def fixed_size(doc, size=40, overlap=0):
    """Cut every `size` words, stepping back `overlap` words each time."""
    words = doc.body.split()
    chunks, start = [], 0
    while start < len(words):
        piece = " ".join(words[start:start + size])
        chunks.append(Chunk(doc.id, len(chunks), "", piece))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def sentences(doc, max_words=60):
    """Pack whole sentences into chunks of at most `max_words` words."""
    text = re.sub(r"^#+ .*$", "", doc.body, flags=re.M)          # drop markdown headings
    sents = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    chunks, current = [], []
    for s in sents:
        if current and len(" ".join(current + [s]).split()) > max_words:
            chunks.append(Chunk(doc.id, len(chunks), "", " ".join(current)))
            current = []
        current.append(s)
    if current:
        chunks.append(Chunk(doc.id, len(chunks), "", " ".join(current)))
    return chunks


def by_heading(doc, max_words=120):
    """One chunk per markdown section; long sections fall back to sentence packing.
    Every chunk starts with the document title and section heading."""
    chunks = []
    sections = re.split(r"^## ", doc.body, flags=re.M)
    intro = re.sub(r"^# .*\n?", "", sections[0]).strip()
    parts = ([("", intro)] if intro else []) + [
        (s.split("\n", 1)[0].strip(), s.split("\n", 1)[1].strip() if "\n" in s else "") for s in sections[1:]
    ]
    for heading, body in parts:
        label = f"{doc.title} > {heading}" if heading else doc.title
        pieces = [body]
        if len(body.split()) > max_words:
            fake = type(doc)(**{**doc.__dict__, "body": body})
            pieces = [c.text for c in sentences(fake, max_words)]
        for p in pieces:
            chunks.append(Chunk(doc.id, len(chunks), label, f"{label}\n{p}"))
    return chunks
