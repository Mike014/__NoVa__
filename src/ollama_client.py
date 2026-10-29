import os
import requests
from dotenv import load_dotenv
from smolagents import ChatMessage, MessageRole, Model
from smolagents.models import get_clean_message_list

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
AGENT_MODEL = os.getenv("OLLAMA_AGENT_MODEL", "llama3.2:3b")


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


class OllamaModel(Model):
    """SmolAgents model on Ollama's native API, so options like OLLAMA_NUM_GPU apply to the agent too
    (the OpenAI-compatible endpoint ignores them)."""

    ROLES = {MessageRole.TOOL_CALL: MessageRole.ASSISTANT, MessageRole.TOOL_RESPONSE: MessageRole.USER}

    def __init__(self, model_id: str = AGENT_MODEL, num_ctx: int = 8192):
        super().__init__(model_id=model_id, flatten_messages_as_text=True)
        self.num_ctx = num_ctx

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        clean = get_clean_message_list(messages, role_conversions=self.ROLES, flatten_messages_as_text=True)
        options = {"num_ctx": self.num_ctx}
        if stop_sequences:
            options["stop"] = stop_sequences
        return ChatMessage(role=MessageRole.ASSISTANT, content=chat(self.model_id, clean, options))


def agent_model(model_id: str = AGENT_MODEL) -> OllamaModel:
    """Returns a SmolAgents model backed by the local Ollama server."""
    return OllamaModel(model_id)
