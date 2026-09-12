"""Willowagent : graphe LangGraph robuste, prêt à être importé par main.py."""

from typing import Literal, TypedDict

import requests
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langgraph.graph import END, START, StateGraph

import database
import exa_tool


load_dotenv()

# --- LLM LOCAL ---
llm = Ollama(model="llama3")


# --- STRUCTURE DE L'ÉTAT ---
class AgentState(TypedDict):
    question: str
    user_role: str  # "etudiant" ou "chercheur"
    user_id: str  # Numéro WhatsApp
    pdf_context: str
    web_context: str
    route: str  # "pdf_search", "web_search", ou "direct"
    final_answer: str


# --- PROMPTS ---
PROMPT_ETUDIANT = """Tu es un tuteur pédagogique bienveillant pour étudiants.
Explique simplement, étape par étape, avec un ton encourageant.
Formate pour WhatsApp : puces claires, gras, émojis légers.

Contexte disponible :
{context}

Question : {question}
Réponse :"""

PROMPT_CHERCHEUR = """Tu es un assistant de recherche scientifique rigoureux.
Fournis une analyse synthétique, précise et axée sur la méthodologie et les sources.
Conserve un ton académique et neutre.

Contexte disponible :
{context}

Question : {question}
Réponse :"""


# --- NŒUDS AVEC GESTION D'ERREURS ---
def router_node(state: AgentState) -> dict[str, str]:
    question = state["question"].lower()
    pdf_keywords = ["pdf", "document", "fichier", "cours", "papier", "l'article", "page", "résume"]
    web_keywords = ["actu", "récent", "recherche", "exa", "actualité", "2024", "2025", "2026", "web"]

    if any(keyword in question for keyword in pdf_keywords):
        route = "pdf_search"
    elif any(keyword in question for keyword in web_keywords):
        route = "web_search"
    else:
        try:
            router_prompt = (
                f"Analyse : '{state['question']}'. Si salutation/question simple -> DIRECT, "
                "si besoin de faits/recherche -> WEB. Réponds avec 1 mot : DIRECT ou WEB."
            )
            decision = str(llm.invoke(router_prompt)).strip().upper()
            route = "web_search" if "WEB" in decision else "direct"
        except Exception:
            # Le graphe reste utilisable même si Ollama est hors ligne pendant le routage.
            route = "direct"
    return {"route": route}


def retrieve_pdf_node(state: AgentState) -> dict[str, str]:
    user_id = state.get("user_id", "")
    question = state["question"]
    try:
        context = database.search_user_pdf(query=question, phone_number=user_id, k=3)
        if not context or not context.strip():
            context = "[Aucun extrait de PDF trouvé pour cette question.]"
    except Exception as exc:
        context = f"[Erreur lors de la recherche PDF : {exc}]"
    return {"pdf_context": context}


def web_search_node(state: AgentState) -> dict[str, str]:
    question = state["question"]
    role = state.get("user_role", "etudiant")
    try:
        context = exa_tool.search_online_research(query=question, user_role=role)
    except Exception as exc:
        # Clé invalide, erreur réseau, limite API : ne jamais interrompre le graphe.
        context = f"[Erreur de recherche Web Exa : {exc}]"
    return {"web_context": context}


def generate_node(state: AgentState) -> dict[str, str]:
    role = state.get("user_role", "etudiant")
    question = state["question"]
    context_parts = []
    if state.get("pdf_context"):
        context_parts.append(f"--- SOURCES PDF ---\n{state['pdf_context']}")
    if state.get("web_context"):
        context_parts.append(f"--- SOURCES WEB (EXA) ---\n{state['web_context']}")
    full_context = "\n\n".join(context_parts) if context_parts else "Aucun document externe."

    template = PROMPT_ETUDIANT if role == "etudiant" else PROMPT_CHERCHEUR
    formatted_prompt = template.format(context=full_context, question=question)
    try:
        final_response = str(llm.invoke(formatted_prompt))
    except requests.exceptions.ConnectionError:
        final_response = "⚠️ Impossible de contacter Ollama local. Vérifiez que `ollama run llama3` est actif."
    except Exception as exc:
        final_response = (
            "⚠️ Impossible de générer la réponse avec Ollama local. "
            f"Vérifiez le serveur et le modèle llama3. Erreur : {exc}"
        )
    return {"final_answer": final_response}


def route_decision(state: AgentState) -> Literal["retrieve_pdf", "web_search", "generate"]:
    if state.get("route") == "pdf_search":
        return "retrieve_pdf"
    if state.get("route") == "web_search":
        return "web_search"
    return "generate"


# --- GRAPHE LANGGRAPH ---
builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("retrieve_pdf", retrieve_pdf_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router",
    route_decision,
    {"retrieve_pdf": "retrieve_pdf", "web_search": "web_search", "generate": "generate"},
)
builder.add_edge("retrieve_pdf", "generate")
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

# Export global : `from agent import agent_app` est supporté par main.py.
agent_app = builder.compile()


def respond(question: str, user_id: str, user_role: str = "etudiant") -> str:
    """Raccourci optionnel conservé pour l'API FastAPI existante."""
    result = agent_app.invoke(
        {
            "question": question,
            "user_role": user_role,
            "user_id": user_id,
            "pdf_context": "",
            "web_context": "",
            "route": "",
            "final_answer": "",
        }
    )
    return result["final_answer"]
