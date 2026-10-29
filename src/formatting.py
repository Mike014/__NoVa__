import re

DISCORD_LIMIT = 2000

# Leftover chat-template tokens some models still print
TOKEN_PATTERNS = (
    r"</?\|?eot_id\|?>",
    r"</?\|?start_header_id\|?>",
    r"<\|end_header_id\|>",
    r"\[/?INST\]",
    r"</?s>",
    r"<</?SYS>>",
    r"<\|.*?\|>",
)

# Imitated chat prefixes like "[mikelskyy]:" or "NoVa:" at the start of the reply
PREFIX = re.compile(r"^\s*(?:\[[^\]\n]{1,40}\]|__?NoVa__?|NoVa|Assistant)\s*:\s*", re.IGNORECASE)
CODE_BLOCK = re.compile(r"```[\s\S]*?```")


def clean_response(text: str) -> str:
    """Removes template tokens and fake prefixes; when the reply has code, keeps only the code blocks."""
    for pattern in TOKEN_PATTERNS:
        text = re.sub(pattern, "", text)
    text = PREFIX.sub("", text).strip()
    blocks = CODE_BLOCK.findall(text)
    if blocks:
        text = "\n\n".join(blocks)
    return text


def split_message(text: str, limit: int = DISCORD_LIMIT) -> list[str]:
    """Splits text into Discord-sized chunks on line breaks, closing and reopening code fences across chunks."""
    chunks, current, fence = [], "", None
    budget = limit - 4  # room for a closing "\n```"

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.rstrip("\n") + ("\n```" if fence else ""))
        current = f"{fence}\n" if fence else ""

    for line in text.split("\n"):
        while len(line) > budget - len(current):
            if current.strip() and current != f"{fence}\n":
                flush()
                continue
            room = budget - len(current)
            current += line[:room]
            line = line[room:]
            flush()
        current += line + "\n"
        if line.strip().startswith("```"):
            fence = None if fence else line.strip()
    current = current.rstrip("\n")
    if current.strip() and current != fence:
        chunks.append(current)
    return chunks
