import os
import requests
from dotenv import load_dotenv
from smolagents import OpenAIServerModel

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "qwen2.5-coder:7b")


def _options(options: dict | None) -> dict:
    """Merges call options with OLLAMA_NUM_GPU from the environment, when set."""
    merged = dict(options or {})
    raw = os.getenv("OLLAMA_NUM_GPU", "").strip()
    if raw != "" and "num_gpu" not in merged:
        merged["num_gpu"] = int(raw)
    return merged


def chat(model: str, messages: list[dict], options: dict | None = None, timeout: int = 300) -> str:
    """Sends a chat request to the local Ollama server and returns the reply text."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": _options(options)},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def agent_model(model_id: str = AGENT_MODEL) -> OpenAIServerModel:
    """Returns a SmolAgents model backed by Ollama's OpenAI-compatible endpoint."""
    return OpenAIServerModel(model_id=model_id, api_base=f"{OLLAMA_HOST}/v1", api_key="ollama")
