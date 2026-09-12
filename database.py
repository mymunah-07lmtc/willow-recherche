"""Accès Supabase : profils WhatsApp, embeddings Hugging Face et pgvector."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from supabase import Client, create_client


load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
ADMIN_RAG_USER_ID = "__admin__"


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Construit le client seulement au premier accès, après validation de .env."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL et SUPABASE_KEY doivent être définies dans .env.")
    return create_client(url, key)


def _get_hf_token() -> str:
    token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_KEY doit être définie dans .env.")
    return token


def get_embedding(text: str) -> list[float]:
    """Retourne un vecteur de 384 dimensions produit par Hugging Face."""
    client = InferenceClient(api_key=_get_hf_token())
    vector = client.feature_extraction(text, model=EMBEDDING_MODEL)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if vector and isinstance(vector[0], list):
        # Certains fournisseurs renvoient un vecteur par token : moyenne simple.
        vector = [sum(values) / len(vector) for values in zip(*vector)]
    result = [float(value) for value in vector]
    if len(result) != 384:
        raise ValueError(f"Embedding inattendu : {len(result)} dimensions au lieu de 384.")
    return result


class VectorStore:
    """Persistance des chunks PDF et recherche vectorielle filtrée par utilisateur."""

    def add_texts(self, texts: list[str], metadatas: list[dict]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts et metadatas doivent avoir la même longueur.")
        rows = [
            {
                "content": text,
                "embedding": get_embedding(text),
                "user_id": metadata["user_id"],
                "page": metadata.get("page"),
            }
            for text, metadata in zip(texts, metadatas)
        ]
        if rows:
            get_supabase_client().table("documents").insert(rows).execute()

    def search(self, query: str, user_id: str, match_count: int = 5) -> list[dict]:
        result = get_supabase_client().rpc(
            "match_documents",
            {
                "query_embedding": get_embedding(query),
                "match_user_id": user_id,
                "match_count": match_count,
            },
        ).execute()
        return result.data or []


def get_vector_store() -> VectorStore:
    return VectorStore()


def search_user_pdf(query: str, phone_number: str, k: int = 5) -> str:
    """Retourne les passages privés puis la documentation RAG partagée admin."""
    store = get_vector_store()
    results = store.search(query=query, user_id=phone_number, match_count=k)
    if phone_number != ADMIN_RAG_USER_ID:
        results.extend(store.search(query=query, user_id=ADMIN_RAG_USER_ID, match_count=k))
    results.sort(key=lambda item: item.get("similarity", 0), reverse=True)
    results = results[:k]
    if not results:
        return "[Aucun extrait de PDF trouvé pour cette question.]"
    passages = []
    for item in results:
        page = item.get("page", "?")
        passages.append(f"[Page {page}] {item.get('content', '')}")
    return "\n\n".join(passages)


def get_user(phone_number: str) -> dict | None:
    result = get_supabase_client().table("users").select("phone_number, role, step").eq(
        "phone_number", phone_number
    ).limit(1).execute()
    return result.data[0] if result.data else None


def create_user(phone_number: str, step: str = "awaiting_role") -> None:
    get_supabase_client().table("users").upsert(
        {"phone_number": phone_number, "step": step}, on_conflict="phone_number"
    ).execute()


def update_user_step(phone_number: str, step: str) -> None:
    get_supabase_client().table("users").update({"step": step}).eq(
        "phone_number", phone_number
    ).execute()


def update_user_role(phone_number: str, role: str) -> None:
    get_supabase_client().table("users").update({"role": role, "step": "active"}).eq(
        "phone_number", phone_number
    ).execute()
