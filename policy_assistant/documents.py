"""Load policy documents: a small front-matter block, then markdown."""
import hashlib
import os
from dataclasses import dataclass, field

from . import config


@dataclass
class Document:
    id: str
    title: str
    category: str
    version: str
    effective: str
    status: str
    audience: str
    body: str
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self):
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:16]


def parse(text):
    _, front, body = text.split("---\n", 2)
    meta = dict(line.split(": ", 1) for line in front.strip().splitlines())
    return Document(meta["id"], meta["title"], meta["category"], meta["version"],
                    meta["effective"], meta["status"], meta["audience"], body.strip(), meta)


def load_all(folder=config.POLICY_DIR):
    docs = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".md"):
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                docs.append(parse(f.read()))
    return docs
