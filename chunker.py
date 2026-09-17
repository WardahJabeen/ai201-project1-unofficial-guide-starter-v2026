"""
Stage 2 of the pipeline: splitting documents into chunks.

⚠️ THIS IS THE FILE YOU CHANGE IN MILESTONE 3.

`split_documents` below is deliberately plain. It cuts every document into
fixed-size pieces with a fixed overlap and pays no attention to where sentences
or paragraphs end. It works, and it is not good.

On a corpus of short posts it may not cut anything at all: `campus_life` comes
out as 88 documents and 88 chunks, because almost nothing in it reaches 800
characters. That is the baseline, not a bug — Milestone 3 is where you decide
whether one post should stay one chunk.

Your job in Milestone 3 is to replace the *body* of `split_documents` with a
strategy that fits the documents you actually read in Milestone 1. Keep the
name and the shape of what it returns — the rest of the pipeline calls it, and
your README has to name the function that produced your chunks.

If you get stuck for 30 minutes, `fallback_split` is the original. Switch back
to it, write down what you saw, and move on. That's a real observation about
your pipeline, not giving up.
"""

import re
from dataclasses import dataclass

import config
from ingest import Document

# Start of a `## ` section. Zero-width, so re.split keeps the heading line
# attached to the body underneath it.
_SECTION_START = re.compile(r"(?m)^(?=##\s)")


@dataclass
class Chunk:
    """One piece of one document."""

    text: str
    source: str        # which file it came from
    index: int         # which chunk within that file, starting at 0
    produced_by: str   # the function that made it — cite this in your README

    @property
    def label(self) -> str:
        return f"{self.source}#{self.index}"


def fallback_split(
    documents: list[Document],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    The starter's original chunker. Fixed-size character windows with overlap.

    Keep this function. Milestone 3's stop rule points back at it, and having
    something to compare your own strategy against is useful in unit 2.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    if overlap >= chunk_size:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []
    for doc in documents:
        start = 0
        index = 0
        while start < len(doc.text):
            piece = doc.text[start : start + chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::fallback_split",
                    )
                )
                index += 1
            start += chunk_size - overlap

    return chunks


def _fit(section: str, max_chars: int, overlap: int) -> list[str]:
    """
    One section, cut only if it is over `max_chars`.

    Cuts fall on paragraph breaks, and each piece after the first opens with
    the last `overlap` characters of the one before it. A section that already
    fits is returned untouched, so the overlap never applies to it.
    """
    if len(section) <= max_chars:
        return [section]

    pieces: list[str] = []
    buffer = ""

    for paragraph in re.split(r"\n\s*\n", section):
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if not buffer:
            buffer = paragraph
        elif len(buffer) + 2 + len(paragraph) <= max_chars:
            buffer = f"{buffer}\n\n{paragraph}"
        else:
            pieces.append(buffer)
            buffer = f"{buffer[-overlap:]}\n\n{paragraph}"

    if buffer:
        pieces.append(buffer)

    bounded: list[str] = []

    for piece in pieces:
        while len(piece) > max_chars:
            bounded.append(piece[:max_chars])
            piece = piece[max_chars - overlap:]

        bounded.append(piece)

    return bounded


def split_documents(documents: list[Document]) -> list[Chunk]:
    """
    Split each document at its `## ` headings, one chunk per section.

    The city_guides guides are all a `# ` title followed by `## ` sections —
    "Getting there", "Eat and drink", "When to go" — and a section is the unit
    a question actually asks about, so it is the unit this cuts on. Each
    heading line stays attached to the body beneath it.

    A section over CHUNK_SIZE is split at paragraph breaks with CHUNK_OVERLAP
    characters carried across; a section that fits is left whole.
    """
    max_chars = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP

    if overlap >= max_chars:
        raise ValueError("overlap has to be smaller than chunk_size")

    chunks: list[Chunk] = []

    for doc in documents:
        index = 0

        sections = [
            s.strip()
            for s in _SECTION_START.split(doc.text)
            if s.strip()
        ]

        # Attach the opening title/summary to the first ## section.
        if len(sections) > 1:
            opening = sections.pop(0)
            sections[0] = f"{opening}\n\n{sections[0]}"

        for section in sections:
            for piece in _fit(section, max_chars, overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        source=doc.source,
                        index=index,
                        produced_by="chunker.py::split_documents",
                    )
                )
                index += 1

    return chunks


def describe(chunks: list[Chunk]) -> str:
    """A one-line summary, printed after indexing."""
    if not chunks:
        return "0 chunks"
    lengths = [len(c.text) for c in chunks]
    return (
        f"{len(chunks)} chunks, "
        f"{sum(lengths) // len(lengths)} characters on average "
        f"(shortest {min(lengths)}, longest {max(lengths)}), "
        f"produced by {chunks[0].produced_by}"
    )


if __name__ == "__main__":
    from ingest import load_documents

    chunks = split_documents(load_documents())
    print(describe(chunks))
