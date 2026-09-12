import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration chargée depuis l'environnement ou le fichier .env."""

    app_name: str = os.getenv("APP_NAME", "Willowagent")
    app_env: str = os.getenv("APP_ENV", "development")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


load_dotenv()
settings = Settings(
    app_name=os.getenv("APP_NAME", "Willowagent"),
    app_env=os.getenv("APP_ENV", "development"),
    ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
