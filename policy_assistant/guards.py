"""Defences against instructions hidden in documents, and against risky answers."""
import re
from dataclasses import replace

SUSPICIOUS = [
    r"ignore (all |any )?(previous|prior|above|earlier) instructions",
    r"disregard (all |any )?(previous|prior|above|your) (instructions|rules)",
    r"\b(note|message|instructions?) (for|to) (ai|llm|language model|chat) (assistants?|models?|bots?)\b",
    r"\bsystem prompt\b",
    r"\byou are now\b",
]
SECRET_REQUEST = re.compile(
    r"(reply|send|share|give|provide|paste|enter|type)\b.{0,60}\b(card number|password|mfa code|one-time code|cvv|pin)\b",
    re.IGNORECASE)


def scan_text(text):
    return [p for p in SUSPICIOUS if re.search(p, text, re.IGNORECASE)]


def strip_suspicious(doc):
    """Drop paragraphs that look like instructions to an AI. Returns (clean_doc, removed_paragraphs)."""
    kept, removed = [], []
    for para in doc.body.split("\n\n"):
        (removed if scan_text(para) else kept).append(para)
    return replace(doc, body="\n\n".join(kept)), removed


def check_answer(answer):
    """Flag answers that ask the employee for secrets: no policy ever needs that."""
    return {"asks_for_secret": bool(SECRET_REQUEST.search(answer))}
