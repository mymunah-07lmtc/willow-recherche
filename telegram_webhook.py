"""Configure et vérifie le webhook Telegram de WillowAgent.

Usage:
    python telegram_webhook.py check
    python telegram_webhook.py set https://votre-tunnel.ngrok-free.dev/telegram/webhook
"""

import os
import sys

import httpx
from dotenv import load_dotenv


load_dotenv()


def api_url(method: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN est absent du fichier .env.")
    return f"https://api.telegram.org/bot{token}/{method}"


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "check"

    if action == "check":
        bot = httpx.get(api_url("getMe"), timeout=15).json()
        hook = httpx.get(api_url("getWebhookInfo"), timeout=15).json()
        if not bot.get("ok") or not hook.get("ok"):
            raise RuntimeError("Telegram a refusé la vérification du bot ou du webhook.")
        print(f"Bot Telegram : @{bot['result']['username']}")
        print(f"Webhook : {hook['result'].get('url') or '(non configuré)'}")
        print(f"Erreurs en attente : {hook['result'].get('pending_update_count', 0)}")
        return

    if action == "set" and len(sys.argv) == 3:
        secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
        if not secret:
            raise RuntimeError("TELEGRAM_WEBHOOK_SECRET est absent du fichier .env.")
        response = httpx.post(
            api_url("setWebhook"),
            json={
                "url": sys.argv[2],
                "secret_token": secret,
                "allowed_updates": ["message"],
            },
            timeout=20,
        )
        data = response.json()
        if not response.is_success or not data.get("ok"):
            raise RuntimeError(f"Configuration Telegram refusée : {data}")
        print("Webhook Telegram configuré avec succès.")
        return

    raise SystemExit("Usage : python telegram_webhook.py [check | set URL]")


if __name__ == "__main__":
    main()
