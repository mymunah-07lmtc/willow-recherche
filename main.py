"""Canaux WhatsApp et Telegram pour WillowAgent."""

import asyncio
import hashlib
import hmac
import json
import os
from collections.abc import Callable

import httpx
import redis.asyncio as redis
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

load_dotenv()

import database
from agent import agent_app
from chunking_pipeline import process_pdf_bytes, process_whatsapp_pdf


app = FastAPI(title="WillowAgent")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN") or os.getenv("VERIFY_TOKEN", "willow_secret_token")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
WHATSAPP_API = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}"
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


async def send_whatsapp_message(recipient: str, text: str) -> None:
    if not WHATSAPP_TOKEN or not PHONE_NUMBER_ID:
        raise RuntimeError("Configuration WhatsApp incomplète dans .env.")
    async with httpx.AsyncClient(timeout=20) as client:
        for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
            response = await client.post(
                f"{WHATSAPP_API}/messages",
                headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
                json={"messaging_product": "whatsapp", "to": recipient, "type": "text", "text": {"body": chunk}},
            )
            response.raise_for_status()


async def send_telegram_message(recipient: str, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN est absent du fichier .env.")
    async with httpx.AsyncClient(timeout=20) as client:
        for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
            response = await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": recipient, "text": chunk},
            )
            response.raise_for_status()


async def download_telegram_document(file_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=30) as client:
        metadata = await client.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id})
        metadata.raise_for_status()
        file_path = metadata.json()["result"]["file_path"]
        response = await client.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}")
        response.raise_for_status()
        return response.content


async def is_duplicate(message_id: str) -> bool:
    if not message_id:
        return False
    try:
        return not await redis_client.set(f"msg:{message_id}", "1", ex=3600, nx=True)
    except Exception as exc:
        print(f"[redis] indisponible, déduplication désactivée : {exc}")
        return False


def is_valid_whatsapp_signature(payload: bytes, signature: str | None) -> bool:
    if not WHATSAPP_APP_SECRET:
        return False
    expected = "sha256=" + hmac.new(WHATSAPP_APP_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return bool(signature) and hmac.compare_digest(expected, signature)


@app.get("/webhook")
async def verify_whatsapp_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge or "")
    raise HTTPException(status_code=403, detail="Token de vérification invalide.")


@app.post("/webhook")
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    raw_payload = await request.body()
    if not is_valid_whatsapp_signature(raw_payload, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=403, detail="Signature WhatsApp invalide.")
    try:
        value = json.loads(raw_payload)["entry"][0]["changes"][0]["value"]
        if "messages" not in value:
            return {"status": "ignored"}
        message = value["messages"][0]
        if await is_duplicate(message.get("id", "")):
            return {"status": "duplicate"}
        phone_number = message["from"]
        background.add_task(process_message, phone_number, phone_number, message, send_whatsapp_message, "whatsapp")
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"[whatsapp] payload invalide : {exc}")
    return {"status": "ok"}


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, background: BackgroundTasks):
    if TELEGRAM_WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Signature Telegram invalide.")
    update = await request.json()
    incoming = update.get("message")
    if not incoming:
        return {"status": "ignored"}
    chat_id = str(incoming["chat"]["id"])
    document = incoming.get("document")
    message = {
        "id": f"telegram:{update.get('update_id', incoming['message_id'])}",
        "type": "document" if document else "text",
        "text": {"body": incoming.get("text", "")},
        "document": {"id": document["file_id"], "mime_type": document.get("mime_type", "")} if document else {},
    }
    if await is_duplicate(message["id"]):
        return {"status": "duplicate"}
    background.add_task(process_message, f"telegram:{chat_id}", chat_id, message, send_telegram_message, "telegram")
    return {"status": "ok"}


async def process_message(
    user_id: str,
    recipient: str,
    message: dict,
    send: Callable[[str, str], object],
    channel: str,
) -> None:
    """Logique de profil, RAG et PDF commune aux deux plateformes."""
    try:
        msg_type = message.get("type")
        user_text = message.get("text", {}).get("body", "").strip()
        user = await asyncio.to_thread(database.get_user, user_id)
        if not user:
            await asyncio.to_thread(database.create_user, user_id, "awaiting_role")
            await send(recipient, "👋 Bienvenue sur WillowAgent !\n\nChoisis ton profil :\n1️⃣ Étudiant\n2️⃣ Chercheur")
            return
        if msg_type == "text" and user_text.lower() in {"/start", "/mode"}:
            await asyncio.to_thread(database.update_user_step, user_id, "awaiting_role")
            await send(recipient, "Choisis ton profil :\n1. Étudiant\n2. Chercheur")
            return
        if user["step"] == "awaiting_role":
            role = {"1": "etudiant", "2": "chercheur"}.get(user_text)
            if not role:
                await send(recipient, "Réponds uniquement par *1* ou *2*.")
                return
            await asyncio.to_thread(database.update_user_role, user_id, role)
            await send(recipient, f"✅ Profil *{role.title()}* activé !")
            return
        if msg_type == "text":
            await send(recipient, "⏳ Je réfléchis…")
            result = await asyncio.to_thread(agent_app.invoke, {"question": user_text, "user_role": user["role"], "user_id": user_id})
            await send(recipient, result.get("final_answer", "❌ Pas de réponse générée."))
        elif msg_type == "document":
            document = message["document"]
            if document.get("mime_type") != "application/pdf":
                await send(recipient, "⚠️ Envoie uniquement un PDF.")
                return
            await send(recipient, "⚙️ Traitement du PDF en cours…")
            if channel == "telegram":
                pdf_bytes = await download_telegram_document(document["id"])
                ok = await asyncio.to_thread(process_pdf_bytes, pdf_bytes, user_id)
            else:
                ok = await asyncio.to_thread(process_whatsapp_pdf, document["id"], user_id)
            await send(recipient, "✅ PDF indexé. Pose maintenant tes questions." if ok else "❌ Impossible de traiter ce PDF.")
        else:
            await send(recipient, "Envoie-moi une question ou un document PDF.")
    except Exception as exc:
        print(f"[process_message:{channel}] erreur : {exc}")


@app.get("/")
async def health():
    return {"status": "ok", "service": "WillowAgent", "channels": ["whatsapp", "telegram"]}
