"""Willowagent : LangGraph avec recherche PDF/Exa et génération Hugging Face."""

import os
from typing import Literal, TypedDict

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import END, START, StateGraph

import database
import exa_tool


load_dotenv()

HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


class AgentState(TypedDict):
    question: str
    user_role: str
    user_id: str
    pdf_context: str
    web_context: str
    route: str
    final_answer: str


PROMPT_ETUDIANT = """Tu es un tuteur pédagogique bienveillant pour étudiants.
Explique simplement, étape par étape, avec un ton encourageant.
Formate pour WhatsApp : puces claires, gras, émojis légers."""

PROMPT_CHERCHEUR = """Tu es un assistant de recherche scientifique rigoureux.
Fournis une analyse synthétique, précise et axée sur la méthodologie et les sources.
Conserve un ton académique et neutre."""


def invoke_huggingface(system_prompt: str, user_prompt: str) -> str:
    """Appelle l'API Hugging Face sans conserver la clé dans le code source."""
    token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_KEY est absente du fichier .env.")

    client = InferenceClient(api_key=token)
    response = client.chat_completion(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=700,
        temperature=0.3,
    )
    return str(response.choices[0].message.content)


def router_node(state: AgentState) -> dict[str, str]:
    question = state["question"].lower()
    pdf_keywords = ["pdf", "document", "fichier", "cours", "papier", "l'article", "page", "résume"]
    web_keywords = ["actu", "récent", "recherche", "exa", "actualité", "2024", "2025", "2026", "web"]
    if any(keyword in question for keyword in pdf_keywords):
        return {"route": "pdf_search"}
    if any(keyword in question for keyword in web_keywords):
        return {"route": "web_search"}
    return {"route": "direct"}


def retrieve_pdf_node(state: AgentState) -> dict[str, str]:
    try:
        context = database.search_user_pdf(
            query=state["question"], phone_number=state.get("user_id", ""), k=3
        )
        if not context or not context.strip():
            context = "[Aucun extrait de PDF trouvé pour cette question.]"
    except Exception as exc:
        context = f"[Erreur lors de la recherche PDF : {exc}]"
    return {"pdf_context": context}


def web_search_node(state: AgentState) -> dict[str, str]:
    try:
        context = exa_tool.search_online_research(
            query=state["question"], user_role=state.get("user_role", "etudiant")
        )
    except Exception as exc:
        context = f"[Erreur de recherche Web Exa : {exc}]"
    return {"web_context": context}


def generate_node(state: AgentState) -> dict[str, str]:
    context_parts = []
    if state.get("pdf_context"):
        context_parts.append(f"--- SOURCES PDF ---\n{state['pdf_context']}")
    if state.get("web_context"):
        context_parts.append(f"--- SOURCES WEB (EXA) ---\n{state['web_context']}")
    context = "\n\n".join(context_parts) or "Aucun document externe."
    system_prompt = PROMPT_ETUDIANT if state.get("user_role") == "etudiant" else PROMPT_CHERCHEUR
    user_prompt = f"""Contexte disponible :
{context}

Question : {state['question']}
Réponse :"""
    try:
        answer = invoke_huggingface(system_prompt, user_prompt)
    except requests.exceptions.ConnectionError:
        answer = "⚠️ Impossible de contacter Hugging Face. Vérifiez votre connexion Internet."
    except Exception as exc:
        answer = f"⚠️ Impossible de générer la réponse Hugging Face : {exc}"
    return {"final_answer": answer}


def route_decision(state: AgentState) -> Literal["retrieve_pdf", "web_search", "generate"]:
    if state.get("route") == "pdf_search":
        return "retrieve_pdf"
    if state.get("route") == "web_search":
        return "web_search"
    return "generate"


builder = StateGraph(AgentState)
builder.add_node("router", router_node)
builder.add_node("retrieve_pdf", retrieve_pdf_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "router")
builder.add_conditional_edges(
    "router", route_decision,
    {"retrieve_pdf": "retrieve_pdf", "web_search": "web_search", "generate": "generate"},
)
builder.add_edge("retrieve_pdf", "generate")
builder.add_edge("web_search", "generate")
builder.add_edge("generate", END)

# Export global compatible avec : from agent import agent_app
agent_app = builder.compile()


def respond(question: str, user_id: str, user_role: str = "etudiant") -> str:
    result = agent_app.invoke(
        {
            "question": question, "user_role": user_role, "user_id": user_id,
            "pdf_context": "", "web_context": "", "route": "", "final_answer": "",
        }
    )
    return result["final_answer"]
