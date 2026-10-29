import re

MODELS = ("gemma", "llama", "mistral")

# Patterns are matched against the original text, so URLs, queries and texts keep their case.
SEARCH = re.compile(r"^!search\s+(.+)", re.IGNORECASE | re.DOTALL)
CONTEXT = re.compile(r"^!context\b", re.IGNORECASE)
SUMMARIZE = re.compile(
    r"\b(?:summarize|shorten|compress|riassumi)\s+(?:this|the following|questo|il seguente)\s+(?:text|testo)\s*:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
VISIT = re.compile(
    r"\b(?:visit|open|analyze|visita|apri|analizza)\s+(?:the\s+|la\s+)?(?:webpage|page|pagina)\s+(\S+)",
    re.IGNORECASE,
)
CHANGE_MODEL = re.compile(rf"^(?:use|switch to|change to|usa|passa a)\s+({'|'.join(MODELS)})\b", re.IGNORECASE)
TIME = re.compile(r"\b(?:what time is it|tell me the time|che or[ea] (?:sono|è|e'))\b", re.IGNORECASE)


def parse_command(user_input):
    """
    Analyzes the user's message and returns the intent and relevant parameters.
    """
    text = user_input.strip()

    if match := SEARCH.match(text):
        return {"intent": "search_web", "parameters": {"query": match.group(1).strip()}}

    if CONTEXT.match(text):
        return {"intent": "check_context", "parameters": {}}

    if match := SUMMARIZE.search(text):
        return {"intent": "summarize_text", "parameters": {"text": match.group(1).strip()}}

    if match := VISIT.search(text):
        return {"intent": "visit_webpage", "parameters": {"url": match.group(1).strip()}}

    if match := CHANGE_MODEL.match(text):
        return {"intent": "change_model", "parameters": {"model": match.group(1).lower()}}

    if TIME.search(text):
        return {"intent": "get_time", "parameters": {}}

    return {"intent": "unknown", "parameters": {}}


# Local test of the parser
if __name__ == "__main__":
    test_messages = [
        "!search Best AI frameworks in 2025",
        "!context",
        "Summarize this text: Artificial intelligence is transforming the world by automating tasks.",
        "Visit webpage https://huggingface.co/Qwen",
        "switch to Gemma",
        "Che ore sono?",
        "Scrivimi un hello world in C++",
    ]
    for msg in test_messages:
        print(f"User: {msg} -> Parsed: {parse_command(msg)}")
