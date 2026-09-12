"""Accès PostgreSQL/Supabase pour les extraits de PDF de Willowagent.

Table attendue : `pdf_chunks(user_id text, content text, metadata jsonb)`.
Supabase est compatible puisque DATABASE_URL est une URL PostgreSQL.
"""

import os
import re

import psycopg2
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("La variable DATABASE_URL est absente du fichier .env.")
    return psycopg2.connect(database_url)


def search_user_pdf(query: str, phone_number: str, k: int = 5) -> str:
    """Recherche des extraits PDF du propriétaire indiqué par son numéro WhatsApp.

    Les noms d'arguments sont volontairement compatibles avec agent.py :
    `query`, `phone_number` et `k`.
    """
    keywords = [word for word in re.findall(r"[\wÀ-ÿ]{4,}", query.lower())][:8]
    if not keywords:
        return "Aucun passage PDF pertinent n'a été trouvé pour cette question."

    conditions = " OR ".join("LOWER(content) LIKE %s" for _ in keywords)
    params = [phone_number, *[f"%{word}%" for word in keywords], k]
    query = f"""
        SELECT content
        FROM pdf_chunks
        WHERE user_id = %s AND ({conditions})
        LIMIT %s
    """
    try:
        with get_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
    except Exception:
        return "Aucun passage PDF disponible : la base de documents est vide ou indisponible."

    if not rows:
        return "Aucun passage PDF pertinent n'a été trouvé pour cette question."
    return "\n\n--- Extrait PDF ---\n".join(row[0] for row in rows)


def get_user(phone_number: str) -> dict | None:
    """Charge le profil WhatsApp depuis la table Supabase `users`."""
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT phone_number, role, step FROM users WHERE phone_number = %s LIMIT 1",
            (phone_number,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"phone_number": row[0], "role": row[1], "step": row[2]}


def create_user(phone_number: str, step: str = "awaiting_role") -> None:
    """Crée un profil minimal si le numéro n'a encore jamais écrit au bot."""
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (phone_number, step)
            VALUES (%s, %s)
            ON CONFLICT (phone_number) DO NOTHING
            """,
            (phone_number, step),
        )
        connection.commit()


def update_user_step(phone_number: str, step: str) -> None:
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute("UPDATE users SET step = %s WHERE phone_number = %s", (step, phone_number))
        connection.commit()


def update_user_role(phone_number: str, role: str) -> None:
    """Enregistre le rôle et active le profil pour les messages suivants."""
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE users SET role = %s, step = 'active' WHERE phone_number = %s",
            (role, phone_number),
        )
        connection.commit()
