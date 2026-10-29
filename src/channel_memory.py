import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_FILE = os.getenv("NOVA_CHAT_DB", "nova_chat_memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id   INTEGER PRIMARY KEY,
    guild_id     INTEGER,
    channel_id   INTEGER NOT NULL,
    channel_name TEXT,
    author_id    INTEGER NOT NULL,
    author_name  TEXT NOT NULL,
    is_bot       INTEGER NOT NULL DEFAULT 0,
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    jump_url     TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages (channel_id, message_id);
CREATE INDEX IF NOT EXISTS idx_messages_guild ON messages (guild_id, message_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='message_id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.message_id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.message_id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.message_id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.message_id, new.content);
END;
"""

# Phrases that mean the user is asking NoVa to look back in the chat history.
RECALL_CUES = (
    "ricord", "quella volta", "l'altra volta", "avevamo", "abbiamo parlato", "abbiamo detto",
    "ci siamo detti", "abbiamo scritto", "ne abbiamo", "ne parlammo", "hai detto", "ho detto",
    "ho usato", "ho utilizzato", "ho scritto", "mi avevi", "ti avevo", "in passato", "in questa chat",
    "remember", "last time", "we talked", "we discussed", "you said", "i said", "did we",
)

STOPWORDS = {
    # Italian
    "che", "chi", "cui", "non", "per", "con", "tra", "fra", "del", "dei", "degli", "della", "delle",
    "dello", "dal", "dai", "dalla", "dalle", "nel", "nei", "nella", "nelle", "sul", "sui", "sulla",
    "una", "uno", "gli", "le", "lo", "la", "il", "di", "da", "in", "su", "ma", "se", "poi", "come",
    "cosa", "quando", "dove", "perché", "perche", "anche", "ancora", "solo", "sono", "sei", "era",
    "eri", "abbiamo", "avevamo", "avete", "hanno", "hai", "ho", "ha", "mi", "ti", "ci", "vi", "si",
    "tu", "io", "noi", "voi", "lui", "lei", "loro", "mio", "mia", "tuo", "tua", "nostro", "questo",
    "questa", "quello", "quella", "quelli", "quelle", "volta", "volte", "ricordi", "ricorda",
    "ricordo", "ricordare", "ricordarti", "ricordarmi", "riesci", "riesco", "posso", "puoi",
    "parlato", "parlare", "detto", "detti", "dette", "dire", "scritto", "scrivere", "usato",
    "usare", "utilizzato", "utilizzare", "fatto", "stato", "siamo", "essere", "parola", "parole",
    "frase", "chat", "canale", "messaggio", "messaggi", "esattamente", "precisamente", "perfetto",
    "sai", "qualcosa", "tutto", "tutti", "molto", "più", "piu", "già", "gia", "allora", "quindi",
    "nova",
    # English
    "the", "and", "you", "are", "was", "were", "what", "when", "where", "who", "why", "how", "that",
    "this", "with", "for", "about", "did", "does", "have", "has", "had", "remember", "time", "said",
    "talked", "discussed", "our", "your", "from", "not", "but", "can", "could", "would", "there",
    "chat", "channel", "word", "exactly",
}


def is_recall_request(text: str) -> bool:
    """True when the message asks about something said earlier in the chat."""
    lowered = text.lower()
    return any(cue in lowered for cue in RECALL_CUES)


def is_this_channel_query(text: str) -> bool:
    """True when the user asks specifically about the current chat/channel."""
    lowered = text.lower()
    return any(p in lowered for p in ("questa chat", "questo canale", "qui in chat", "this chat", "this channel"))


def to_record(message) -> dict | None:
    """Converts a discord.Message into a row for the messages table, or None if it has no text."""
    if message.type.name not in ("default", "reply"):
        return None
    content = (message.clean_content or "").strip()
    if message.author.bot:
        content = re.sub(r"^\s*\[[^\]\n]{1,40}\]\s*:\s*", "", content)
    if message.attachments:
        names = ", ".join(attachment.filename for attachment in message.attachments)
        content = f"{content} [allegati: {names}]".strip()
    if not content:
        return None
    return {
        "message_id": message.id,
        "guild_id": message.guild.id if message.guild else None,
        "channel_id": message.channel.id,
        "channel_name": getattr(message.channel, "name", None) or "DM",
        "author_id": message.author.id,
        "author_name": message.author.display_name,
        "is_bot": int(message.author.bot),
        "content": content,
        "created_at": message.created_at.isoformat(),
        "jump_url": message.jump_url,
    }


def fts_query(text: str, max_terms: int = 12) -> str:
    """Builds an FTS5 OR-query of prefix terms; roughly stems words so 'crashava' matches 'crash'."""
    words = [w.strip("_") for w in re.findall(r"\w+", text.lower())]
    terms = []
    for word in words:
        if len(word) < 3 or word in STOPWORDS or word.isdigit():
            continue
        stem = word if len(word) <= 5 else word[: max(5, len(word) - 3)]
        terms.append(stem)
    unique = list(dict.fromkeys(terms))[:max_terms]
    return " OR ".join(f'"{term}"*' for term in unique)


def format_date(iso_value: str) -> str:
    return datetime.fromisoformat(iso_value).astimezone().strftime("%d/%m/%Y %H:%M")


def shorten(text: str, limit: int = 400) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_snippets(snippets: list[list[sqlite3.Row]]) -> str:
    """Renders recalled conversation fragments as plain text for the system prompt."""
    blocks = []
    for rows in snippets:
        first = rows[0]
        header = f"[#{first['channel_name']}, {format_date(first['created_at'])}, {first['jump_url']}]"
        lines = [f"{row['author_name']}: {shorten(row['content'])}" for row in rows]
        blocks.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)


class ChatMemory:
    """Persistent archive of Discord messages with full-text recall."""

    def __init__(self, path: str = DB_FILE):
        self.path = path
        with self._db() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def store(self, records: list[dict]) -> int:
        """Saves messages, skipping ones already stored. Returns how many were new."""
        if not records:
            return 0
        with self._db() as conn:
            cursor = conn.executemany(
                """
                INSERT OR IGNORE INTO messages
                    (message_id, guild_id, channel_id, channel_name, author_id, author_name,
                     is_bot, content, created_at, jump_url)
                VALUES
                    (:message_id, :guild_id, :channel_id, :channel_name, :author_id, :author_name,
                     :is_bot, :content, :created_at, :jump_url)
                """,
                records,
            )
            return cursor.rowcount

    def update_content(self, message_id: int, content: str) -> None:
        with self._db() as conn:
            conn.execute("UPDATE messages SET content = ? WHERE message_id = ?", (content, message_id))

    def delete(self, message_id: int) -> None:
        with self._db() as conn:
            conn.execute("DELETE FROM messages WHERE message_id = ?", (message_id,))

    def recent(self, channel_id: int, limit: int = 20) -> list[sqlite3.Row]:
        """Latest messages of a channel, oldest first."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE channel_id = ? ORDER BY message_id DESC LIMIT ?",
                (channel_id, limit),
            ).fetchall()
        return list(reversed(rows))

    def search(self, query: str, guild_id: int | None, channel_id: int,
               exclude_ids: set[int] = frozenset(), limit: int = 4) -> list[sqlite3.Row]:
        """Best-matching human messages in the same server (or the same DM channel)."""
        match = fts_query(query)
        if not match:
            return []
        scope_sql, scope_value = ("m.guild_id = ?", guild_id) if guild_id is not None else ("m.channel_id = ?", channel_id)
        with self._db() as conn:
            rows = conn.execute(
                f"""
                SELECT m.* FROM messages_fts
                JOIN messages m ON m.message_id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND {scope_sql} AND m.is_bot = 0
                ORDER BY bm25(messages_fts)
                LIMIT ?
                """,
                (match, scope_value, limit + len(exclude_ids)),
            ).fetchall()
        return [
            row for row in rows
            if row["message_id"] not in exclude_ids and not is_recall_request(row["content"])
        ][:limit]

    def around(self, channel_id: int, message_id: int, window: int = 2) -> list[sqlite3.Row]:
        """A message with a few human messages before and after it, in chronological order."""
        with self._db() as conn:
            before = conn.execute(
                "SELECT * FROM messages WHERE channel_id = ? AND is_bot = 0 AND message_id < ? ORDER BY message_id DESC LIMIT ?",
                (channel_id, message_id, window),
            ).fetchall()
            center = conn.execute("SELECT * FROM messages WHERE message_id = ?", (message_id,)).fetchall()
            after = conn.execute(
                "SELECT * FROM messages WHERE channel_id = ? AND is_bot = 0 AND message_id > ? ORDER BY message_id ASC LIMIT ?",
                (channel_id, message_id, window),
            ).fetchall()
        return list(reversed(before)) + center + after

    def recent_highlights(self, guild_id: int | None, channel_id: int,
                          exclude_ids: set[int] = frozenset(), limit: int = 8) -> list[sqlite3.Row]:
        """Latest distinct human statements across the server, used when the user asks a broad 'what do you remember?'."""
        scope_sql, scope_value = ("guild_id = ?", guild_id) if guild_id is not None else ("channel_id = ?", channel_id)
        with self._db() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM messages
                WHERE {scope_sql} AND is_bot = 0
                ORDER BY message_id DESC
                LIMIT 200
                """,
                (scope_value,),
            ).fetchall()
        skip_prefixes = ("ciao", "hello", "hi ", "hey", "hei", "salve", "use ", "switch ", "check", "!check")
        picked, seen_text = [], set()
        for row in rows:
            if row["message_id"] in exclude_ids:
                continue
            cleaned = re.sub(r"^@\S+\s*", "", row["content"]).strip()
            key = cleaned.lower()
            if len(cleaned) < 22 or key in seen_text or cleaned.endswith("?"):
                continue
            if key.startswith(skip_prefixes) or is_recall_request(cleaned):
                continue
            seen_text.add(key)
            picked.append(row)
            if len(picked) >= limit:
                break
        return list(reversed(picked))

    def recall(self, query: str, guild_id: int | None, channel_id: int,
               exclude_ids: set[int] = frozenset(), hits: int = 4, window: int = 2) -> list[list[sqlite3.Row]]:
        """Conversation fragments around the messages that best match the query, oldest first."""
        if not fts_query(query):
            if is_this_channel_query(query):
                rows = [r for r in self.recent(channel_id, 16) if r["message_id"] not in exclude_ids]
                return [rows] if rows else []
            highlights = self.recent_highlights(guild_id, channel_id, exclude_ids)
            return [[row] for row in highlights]
        snippets, seen = [], set(exclude_ids)
        for hit in self.search(query, guild_id, channel_id, exclude_ids, hits):
            if hit["message_id"] in seen:
                continue
            rows = [row for row in self.around(hit["channel_id"], hit["message_id"], window)
                    if row["message_id"] not in seen]
            seen.update(row["message_id"] for row in rows)
            if rows:
                snippets.append(rows)
        return sorted(snippets, key=lambda rows: rows[0]["message_id"])
