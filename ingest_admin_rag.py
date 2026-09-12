"""Indexe les PDF du dossier RAG comme documentation partagée des administrateurs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PyPDF2 import PdfReader

import database
from chunking_pipeline import custom_chunking


RAG_DIRECTORY = Path("RAG")


def extract_pages(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    return [
        {"page": number, "text": page.extract_text() or ""}
        for number, page in enumerate(reader.pages, start=1)
    ]


def file_hash(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def ingest_admin_rag() -> tuple[int, int]:
    """Indexe chaque PDF unique; retourne (documents, chunks)."""
    if not RAG_DIRECTORY.exists():
        raise FileNotFoundError(f"Dossier introuvable : {RAG_DIRECTORY.resolve()}")

    store = database.get_vector_store()
    seen_hashes: set[str] = set()
    document_count = chunk_count = 0

    for pdf_path in sorted(RAG_DIRECTORY.glob("*.pdf")):
        digest = file_hash(pdf_path)
        if digest in seen_hashes:
            print(f"Doublon ignoré : {pdf_path.name}")
            continue
        seen_hashes.add(digest)

        chunks = custom_chunking(extract_pages(pdf_path))
        if not chunks:
            print(f"Aucun texte exploitable : {pdf_path.name}")
            continue

        store.add_texts(
            texts=[chunk["text"] for chunk in chunks],
            metadatas=[
                {"user_id": database.ADMIN_RAG_USER_ID, "page": chunk["page"]}
                for chunk in chunks
            ],
        )
        document_count += 1
        chunk_count += len(chunks)
        print(f"Indexé : {pdf_path.name} ({len(chunks)} chunks)")

    return document_count, chunk_count


if __name__ == "__main__":
    documents, chunks = ingest_admin_rag()
    print(f"Terminé : {documents} PDF uniques, {chunks} chunks partagés.")
