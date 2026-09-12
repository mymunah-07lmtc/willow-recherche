"""
chunking_pipeline.py — Personne 3 : ingestion PDF WhatsApp, chunking sur-mesure,
embeddings Ollama, stockage Supabase pgvector avec isolation par user_id.

pip install requests PyPDF2 supabase
(PyPDF2 est dépréciée au profit de pypdf, à l'API identique — `pip install pypdf`
et `from pypdf import PdfReader` fonctionnent aussi si besoin.)
"""

import os
import requests
from io import BytesIO
import PyPDF2
from dotenv import load_dotenv
import database

load_dotenv()
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")


# 1. Téléchargement du fichier depuis l'API WhatsApp
def download_whatsapp_media(media_id: str) -> bytes:
    """Récupère le fichier binaire PDF depuis les serveurs de WhatsApp."""
    if not WHATSAPP_TOKEN:
        raise RuntimeError("WHATSAPP_TOKEN est absent du fichier .env.")
    url_info = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

    # Étape A : récupérer l'URL de téléchargement temporaire
    res_info = requests.get(url_info, headers=headers, timeout=15)
    res_info.raise_for_status()
    download_url = res_info.json().get("url")
    if not download_url:
        raise ValueError(f"Impossible de récupérer l'URL du média {media_id}")

    # Étape B : télécharger le contenu binaire (le token est requis ici aussi)
    res_file = requests.get(download_url, headers=headers, timeout=30)
    res_file.raise_for_status()
    return res_file.content


def _split_long_text(text: str, max_size: int) -> list[str]:
    """Découpe un texte trop long en morceaux de taille fixe, en secours quand
    un paragraphe entier dépasse déjà max_chunk_size (PDF dense sans \\n\\n)."""
    if len(text) <= max_size:
        return [text]
    return [text[i : i + max_size] for i in range(0, len(text), max_size)]


# 2. Algorithme de chunking sur-mesure
def custom_chunking(
    pages_text: list[dict], max_chunk_size: int = 1000, overlap: int = 150
) -> list[dict]:
    """
    Découpe le texte par paragraphes tout en conservant le contexte et la page
    d'origine. pages_text = [{"page": 1, "text": "..."}, ...]
    """
    chunks = []

    for page_data in pages_text:
        page_num = page_data["page"]
        text = page_data["text"]

        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Sécurité : si un seul paragraphe dépasse déjà la taille max
            # (fréquent sur un PDF académique dense), on le découpe directement.
            if len(para) > max_chunk_size:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "page": page_num})
                    current_chunk = ""
                for piece in _split_long_text(para, max_chunk_size):
                    chunks.append({"text": piece.strip(), "page": page_num})
                continue

            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += " " + para
            else:
                if current_chunk:
                    chunks.append({"text": current_chunk.strip(), "page": page_num})
                # Chevauchement pour ne pas perdre la fin de l'idée précédente
                current_chunk = (
                    current_chunk[-overlap:] + " " + para
                    if len(current_chunk) >= overlap
                    else para
                )

        if current_chunk:
            chunks.append({"text": current_chunk.strip(), "page": page_num})

    return chunks


# 3. Pipeline complet d'ingestion, à appeler depuis main.py
def process_whatsapp_pdf(media_id: str, phone_number: str) -> bool:
    try:
        # A. Téléchargement
        pdf_bytes = download_whatsapp_media(media_id)

        # B. Extraction du texte page par page
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pages_text = []
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            pages_text.append({"page": i + 1, "text": text})

        # C. Découpage sur-mesure
        chunked_docs = custom_chunking(pages_text)

        # D. Préparation des textes et métadonnées pour Supabase
        texts_to_embed = [doc["text"] for doc in chunked_docs]
        if not texts_to_embed:
            raise ValueError("Le PDF ne contient aucun texte exploitable.")
        metadatas = [
            {"user_id": phone_number, "page": doc["page"]} for doc in chunked_docs
        ]

        # E. Vectorisation (Ollama) + sauvegarde (Supabase pgvector)
        vector_store = database.get_vector_store()
        vector_store.add_texts(texts=texts_to_embed, metadatas=metadatas)

        return True
    except Exception as e:
        print(f"Erreur lors de l'ingestion du PDF : {e}")
        return False


# 4. Recherche sécurisée — isolée par utilisateur (tâche 5 de la feuille de route)
def search_user_pdf(query: str, phone_number: str, top_k: int = 5) -> list[dict]:
    """
    Recherche par similarité cosinus dans les PDF déjà ingérés, strictement
    filtrée sur le numéro WhatsApp appelant (phone_number = user_id).
    Retourne une liste de {"content", "page", "similarity"}.
    """
    vector_store = database.get_vector_store()
    return vector_store.search(query=query, user_id=phone_number, match_count=top_k)
