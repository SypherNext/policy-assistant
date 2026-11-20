from policy_assistant import chunking
from policy_assistant.documents import load_all

docs = {d.id: d for d in load_all()}
doc = docs["remote-work-policy"]
print(f"{doc.id}: {len(doc.body.split())} words\n")

strategies = {
    "fixed 40 words, no overlap": lambda d: chunking.fixed_size(d, 40, 0),
    "fixed 40 words, 10 overlap": lambda d: chunking.fixed_size(d, 40, 10),
    "sentences, max 60 words":    lambda d: chunking.sentences(d, 60),
    "by heading, max 120 words":  lambda d: chunking.by_heading(d, 120),
}
for name, fn in strategies.items():
    chunks = fn(doc)
    sizes = [len(c.text.split()) for c in chunks]
    print(f"{name:28s} {len(chunks):2d} chunks, words per chunk {min(sizes)} to {max(sizes)}")

print("\nFixed 40 words, no overlap: how each chunk starts and ends")
for c in chunking.fixed_size(doc, 40, 0):
    w = c.text.split()
    print(f"  chunk {c.chunk_no}: {' '.join(w[:4])} ... {' '.join(w[-5:])}")

print("\nBy heading: one chunk per section, title and heading on top")
for c in chunking.by_heading(doc):
    print(f"  chunk {c.chunk_no}: {c.heading}  ({len(c.text.split())} words)")

print("\nWhole corpus (30 documents):")
for name, fn in strategies.items():
    total = sum(len(fn(d)) for d in docs.values())
    print(f"  {name:28s} {total:3d} chunks")
