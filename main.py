"""Webhook WhatsApp asynchrone pour WillowAgent."""

import asyncio
import hashlib
import hmac
import json
import os

import httpx
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

import database
from agent import agent_app
from chunking_pipeline import process_whatsapp_pdf


app = FastAPI(title="WillowAgent WhatsApp Backend")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN") or os.getenv("VERIFY_TOKEN", "willow_secret_token")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WHATSAPP_API = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def send_whatsapp_message(to_phone: str, text: str) -> None:
    """Envoie une réponse, en découpant les messages trop longs pour WhatsApp."""
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("Configuration WhatsApp incomplète dans .env.")
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    async with httpx.AsyncClient(timeout=20) as client:
        for chunk in (text[index:index + 4000] for index in range(0, len(text), 4000)):
            response = await client.post(
                f"{WHATSAPP_API}/messages",
                headers=headers,
                json={
                    "messaging_product": "whatsapp",
                    "to": to_phone,
                    "type": "text",
                    "text": {"body": chunk},
                },
            )
            response.raise_for_status()


async def is_duplicate(message_id: str) -> bool:
    """Évite les doublons WhatsApp; le bot reste fonctionnel sans Redis."""
    if not message_id:
        return False
    try:
        return not await redis_client.set(f"msg:{message_id}", "1", ex=3600, nx=True)
    except Exception as exc:
        print(f"[redis] indisponible, déduplication désactivée : {exc}")
        return False


def is_valid_webhook_signature(payload: bytes, signature: str | None) -> bool:
    if not WHATSAPP_APP_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        WHATSAPP_APP_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return bool(signature) and hmac.compare_digest(expected, signature)


@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge or "")
    raise HTTPException(status_code=403, detail="Token de vérification invalide.")


@app.post("/webhook")
async def handle_webhook(request: Request, background: BackgroundTasks):
    raw_payload = await request.body()
    if not is_valid_webhook_signature(raw_payload, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=403, detail="Signature WhatsApp invalide.")
    try:
        value = json.loads(raw_payload)["entry"][0]["changes"][0]["value"]
        if "messages" not in value:
            return {"status": "ignored"}
        message = value["messages"][0]
        if await is_duplicate(message.get("id", "")):
            return {"status": "duplicate"}
        background.add_task(process_message, message["from"], message)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[webhook] payload invalide : {exc}")
    return {"status": "ok"}


async def process_message(phone_number: str, message: dict) -> None:
    """Orchestre le profil utilisateur, l'agent et l'ingestion PDF hors réponse webhook."""
    try:
        msg_type = message.get("type")
        user_text = message.get("text", {}).get("body", "").strip()
        user = await asyncio.to_thread(database.get_user, phone_number)

        if not user:
            await asyncio.to_thread(database.create_user, phone_number, "awaiting_role")
            await send_whatsapp_message(
                phone_number,
                "👋 Bienvenue sur WillowAgent !\n\nChoisis ton profil :\n"
                "1️⃣ Étudiant\n2️⃣ Chercheur",
            )
            return

        if msg_type == "text" and user_text.lower() == "/mode":
            await asyncio.to_thread(database.update_user_step, phone_number, "awaiting_role")
            await send_whatsapp_message(phone_number, "Choisis ton nouveau profil :\n1. Étudiant\n2. Chercheur")
            return

        if user["step"] == "awaiting_role":
            role = {"1": "etudiant", "2": "chercheur"}.get(user_text)
            if not role:
                await send_whatsapp_message(phone_number, "Réponds uniquement par *1* ou *2*.")
                return
            await asyncio.to_thread(database.update_user_role, phone_number, role)
            await send_whatsapp_message(phone_number, f"✅ Profil *{role.title()}* activé !")
            return

        if msg_type == "text":
            await send_whatsapp_message(phone_number, "⏳ Je réfléchis…")
            result = await asyncio.to_thread(
                agent_app.invoke,
                {"question": user_text, "user_role": user["role"], "user_id": phone_number},
            )
            await send_whatsapp_message(phone_number, result.get("final_answer", "❌ Pas de réponse générée."))
        elif msg_type == "document":
            document = message.get("document", {})
            if document.get("mime_type") != "application/pdf":
                await send_whatsapp_message(phone_number, "⚠️ Envoie uniquement un PDF.")
                return
            await send_whatsapp_message(phone_number, "⚙️ Traitement du PDF en cours…")
            ok = await asyncio.to_thread(process_whatsapp_pdf, document["id"], phone_number)
            await send_whatsapp_message(
                phone_number,
                "✅ PDF indexé. Pose maintenant tes questions." if ok else "❌ Impossible de traiter ce PDF.",
            )
        else:
            await send_whatsapp_message(phone_number, "Envoie-moi une question ou un document PDF.")
    except Exception as exc:
        print(f"[process_message] erreur : {exc}")


@app.get("/")
async def health():
    return {"status": "ok", "service": "WillowAgent"}
