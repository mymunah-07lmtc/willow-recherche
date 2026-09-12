import os
import hashlib
import hmac
import json
import requests
from fastapi import FastAPI, Request, HTTPException, Query
from dotenv import load_dotenv

load_dotenv()

import database
from agent import agent_app  # Développé par la Personne 1
from chunking_pipeline import process_whatsapp_pdf

app = FastAPI(title="WillowAgent WhatsApp Backend")

WHATSAPP_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN") or os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN") or os.getenv("VERIFY_TOKEN", "willow_secret_token")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "")

# --- HELPER : ENVOI WHATSAPP ---
def send_whatsapp_message(to_phone: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text}
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def is_valid_webhook_signature(payload: bytes, signature: str | None) -> bool:
    """Valide la signature Meta pour empêcher les requêtes webhook forgées."""
    if not WHATSAPP_APP_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        WHATSAPP_APP_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return bool(signature) and hmac.compare_digest(expected, signature)

# --- VERIFICATION WEBHOOK (GET) ---
@app.get("/webhook")
async def verify_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge")
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    raise HTTPException(status_code=403, detail="Token de vérification invalide.")

# --- RECEPTION MESSAGES (POST) ---
@app.post("/webhook")
async def handle_webhook(request: Request):
    raw_payload = await request.body()
    if not is_valid_webhook_signature(
        raw_payload, request.headers.get("X-Hub-Signature-256")
    ):
        raise HTTPException(status_code=403, detail="Signature WhatsApp invalide.")
    data = json.loads(raw_payload)
    
    try:
        # Extraction basique du payload WhatsApp Meta
        entry = data["entry"][0]["changes"][0]["value"]
        if "messages" not in entry:
            return {"status": "ignored"}
            
        message = entry["messages"][0]
        phone_number = message["from"]
        msg_type = message.get("type")
        
        # 1. Vérifier / Charger l'utilisateur depuis Supabase
        user = database.get_user(phone_number)
        
        # 2. Nouvel Utilisateur
        if not user:
            database.create_user(phone_number, step="awaiting_role")
            welcome_msg = (
                "👋 Bienvenue sur WillowAgent !\n\n"
                "Pour adapter mes réponses, quel est votre profil ?\n"
                "1️⃣ Étudiant (explications simples, fiches, quiz)\n"
                "2️⃣ Chercheur (analyse rigoureuse, sources académiques)"
            )
            send_whatsapp_message(phone_number, welcome_msg)
            return {"status": "ok"}
            
        # 3. Commande spéciale pour changer de mode
        user_text = message.get("text", {}).get("body", "").strip()
        if user_text.lower() == "/mode":
            database.update_user_step(phone_number, step="awaiting_role")
            send_whatsapp_message(phone_number, "Choisissez votre nouveau profil :\n1. Étudiant\n2. Chercheur")
            return {"status": "ok"}

        # 4. En attente du choix de rôle
        if user["step"] == "awaiting_role":
            if user_text == "1":
                database.update_user_role(phone_number, role="etudiant")
                send_whatsapp_message(phone_number, "✅ Profil **Étudiant** activé ! Envoie un PDF ou pose ta question.")
            elif user_text == "2":
                database.update_user_role(phone_number, role="chercheur")
                send_whatsapp_message(phone_number, "✅ Profil **Chercheur** activé ! Pose tes questions scientifiques ou envoie un document.")
            else:
                send_whatsapp_message(phone_number, "Veuillez répondre uniquement par '1' (Étudiant) ou '2' (Chercheur).")
            return {"status": "ok"}

        # 5. Utilisateur Actif -> Traitement de la requête
        if user["step"] == "active":
            if msg_type == "text":
                # Appeler l'Agent LangGraph
                inputs = {
                    "question": user_text,
                    "user_role": user["role"],
                    "user_id": phone_number
                }
                result = agent_app.invoke(inputs)
                send_whatsapp_message(phone_number, result["final_answer"])
                
            elif msg_type == "document":
                send_whatsapp_message(phone_number, "⚙️ Traitement de ton PDF en cours...")
                document = message.get("document", {})
                if document.get("mime_type") != "application/pdf":
                    send_whatsapp_message(phone_number, "Envoie-moi un document au format PDF.")
                elif process_whatsapp_pdf(document["id"], phone_number):
                    send_whatsapp_message(phone_number, "✅ PDF indexé. Tu peux maintenant poser tes questions.")
                else:
                    send_whatsapp_message(phone_number, "⚠️ Je n'ai pas pu traiter ce PDF. Réessaie plus tard.")

    except Exception as e:
        print(f"Erreur Webhook: {e}")
        
    return {"status": "ok"}


