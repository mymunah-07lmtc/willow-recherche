"""Recherche en ligne via Exa AI."""

import os

from dotenv import load_dotenv
from exa_py import Exa


load_dotenv()


def search_online_research(question: str, user_role: str, num_results: int = 5) -> str:
    """Retourne un contexte Web adapté à un profil étudiant ou chercheur."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key or api_key == "TA_CLE_EXA_ICI":
        return "Recherche Web indisponible : configurez EXA_API_KEY dans le fichier .env."

    query = question
    if user_role == "chercheur":
        query = f"{question} articles scientifiques recherche académique"

    try:
        results = Exa(api_key=api_key).search_and_contents(
            query,
            num_results=num_results,
            text={"max_characters": 1_500},
        )
    except Exception as exc:
        return f"Recherche Web indisponible : {exc}"

    context = []
    for item in results.results:
        context.append(
            f"Titre : {item.title or 'Sans titre'}\n"
            f"URL : {item.url}\n"
            f"Contenu : {item.text or 'Contenu non disponible'}"
        )
    return "\n\n---\n\n".join(context) or "Aucun résultat Web pertinent n'a été trouvé."
