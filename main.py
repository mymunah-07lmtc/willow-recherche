import os
import asyncio
import httpx
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

import database
from agent import agent_app
from ingestion import ingest_pdf  # géré par P3

load_dotenv()

app = FastAPI(title="WillowAgent WhatsApp Backend")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "willow_secret_token")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

r = redis.from_url(REDIS_URL, decode_responses=True)

WHATSAPP_API = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}"


# ============================================================
# HELPERS
# ============================================================

async def send_whatsapp_message(to_phone: str, text: str):
    """Envoie un message texte, découpé si > 4000 chars."""
    if not text:
        return
    url = f"{WHATSAPP_API}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    chunks = [text[i:i + 4000] for i in range(0, len(text), 4000)]

    async with httpx.AsyncClient(timeout=15) as client:
        for chunk in chunks:
            try:
                await client.post(url, headers=headers, json={
                    "messaging_product": "whatsapp",
                    "to": to_phone,
                    "type": "text",
                    "text": {"body": chunk}
                })
            except Exception as e:
                print(f"[send_whatsapp] erreur: {e}")


async def download_whatsapp_media(media_id: str) -> bytes:
    """Récupère un média (PDF) depuis WhatsApp."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient(timeout=30) as client:
        # 1. Obtenir l'URL du média
        meta_res = await client.get(
            f"https://graph.facebook.com/v18.0/{media_id}",
            headers=headers
        )
        meta = meta_res.json()
        media_url = meta.get("url")
        if not media_url:
            raise ValueError(f"Média introuvable: {meta}")

        # 2. Télécharger le binaire
        file_res = await client.get(media_url, headers=headers)
        return file_res.content


async def is_duplicate(message_id: str) -> bool:
    """Empêche le double traitement (WhatsApp retry)."""
    key = f"msg:{message_id}"
    # SETNX retourne True si la clé n'existait pas
    created = await r.set(key, "1", ex=3600, nx=True)
    return not created


# ============================================================
# WEBHOOK VERIFICATION (GET)
# ============================================================

@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="Token invalide.")


# ============================================================
# WEBHOOK RECEPTION (POST)
# ============================================================

@app.post("/webhook")
async def handle_webhook(request: Request, background: BackgroundTasks):
    try:
        data = await request.json()
        entry = data["entry"][0]["changes"][0]["value"]

        # Ignore les events "statuses" (delivered, read…)
        if "messages" not in entry:
            return {"status": "ignored"}

        message = entry["messages"][0]
        message_id = message.get("id", "")
        phone_number = message["from"]

        # Dédup
        if await is_duplicate(message_id):
            return {"status": "duplicate"}

        # Traiter en arrière-plan pour répondre <1s à Meta
        background.add_task(process_message, phone_number, message)

    except Exception as e:
        print(f"[webhook] erreur: {e}")

    return {"status": "ok"}


# ============================================================
# LOGIQUE MÉTIER (background)
# ============================================================

async def process_message(phone_number: str, message: dict):
    try:
        msg_type = message.get("type")
        user_text = message.get("text", {}).get("body", "").strip()

        user = database.get_user(phone_number)

        # ---------- 1. NOUVEL UTILISATEUR ----------
        if not user:
            database.create_user(phone_number, step="awaiting_role")
            await send_whatsapp_message(phone_number,
                "👋 Bienvenue sur WillowAgent !\n\n"
                "Pour adapter mes réponses, quel est ton profil ?\n"
                "1️⃣ Étudiant (explications simples, fiches, quiz)\n"
                "2️⃣ Chercheur (analyse rigoureuse, sources académiques)"
            )
            return

        # ---------- 2. COMMANDE /mode ----------
        if msg_type == "text" and user_text.lower() == "/mode":
            database.update_user_step(phone_number, step="awaiting_role")
            await send_whatsapp_message(phone_number,
                "Choisis ton nouveau profil :\n1. 🎓 Étudiant\n2. 🔬 Chercheur"
            )
            return

        # ---------- 3. EN ATTENTE DU RÔLE ----------
        if user["step"] == "awaiting_role":
            if user_text == "1":
                database.update_user_role(phone_number, role="etudiant")
                await send_whatsapp_message(phone_number,
                    "✅ Profil *Étudiant* activé ! Envoie un PDF ou pose ta question."
                )
            elif user_text == "2":
                database.update_user_role(phone_number, role="chercheur")
                await send_whatsapp_message(phone_number,
                    "✅ Profil *Chercheur* activé ! Pose ta question ou envoie un document."
                )
            else:
                await send_whatsapp_message(phone_number,
                    "Réponds uniquement par *1* (Étudiant) ou *2* (Chercheur)."
                )
            return

        # ---------- 4. UTILISATEUR ACTIF ----------
        if user["step"] != "active":
            return

        # --- 4a. TEXTE → Agent ---
        if msg_type == "text":
            await send_whatsapp_message(phone_number, "⏳ Je réfléchis…")
            try:
                result = await asyncio.to_thread(
                    agent_app.invoke,
                    {
                        "question": user_text,
                        "user_role": user["role"],
                        "user_id": phone_number
                    }
                )
                answer = result.get("final_answer", "❌ Pas de réponse générée.")
                await send_whatsapp_message(phone_number, answer)
            except Exception as e:
                print(f"[agent] erreur: {e}")
                await send_whatsapp_message(phone_number,
                    "❌ Erreur pendant le traitement. Réessaie."
                )

        # --- 4b. DOCUMENT PDF → Ingestion ---
        elif msg_type == "document":
            doc = message.get("document", {})
            mime = doc.get("mime_type", "")
            filename = doc.get("filename", "document.pdf")
            media_id = doc.get("id")

            if "pdf" not in mime.lower():
                await send_whatsapp_message(phone_number,
                    "⚠️ Envoie uniquement des fichiers PDF pour l'instant."
                )
                return

            await send_whatsapp_message(phone_number,
                "⚙️ Traitement de ton PDF en cours…"
            )
            try:
                pdf_bytes = await download_whatsapp_media(media_id)
                doc_id = await asyncio.to_thread(
                    ingest_pdf, phone_number, pdf_bytes, filename
                )
                await send_whatsapp_message(phone_number,
                    f"📄 Document *{filename}* reçu et indexé !\n"
                    "Tu peux maintenant me poser tes questions."
                )
            except Exception as e:
                print(f"[ingestion] erreur: {e}")
                await send_whatsapp_message(phone_number,
                    "❌ Impossible de traiter ce PDF. Réessaie."
                )

        # --- 4c. AUDIO (optionnel) ---
        elif msg_type == "audio":
            await send_whatsapp_message(phone_number,
                "🎙️ Notes vocales bientôt disponibles."
            )

        # --- 4d. AUTRE ---
        else:
            await send_whatsapp_message(phone_number,
                "Je gère pour l'instant : texte, PDF. Envoie-moi l'un des deux."
            )

    except Exception as e:
        print(f"[process_message] erreur: {e}")


# ============================================================
# HEALTHCHECK
# ============================================================

@app.get("/")
async def health():
    return {"status": "ok", "service": "WillowAgent"}
