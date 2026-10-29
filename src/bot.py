import asyncio
import discord
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from ollama_client import chat
from parser import parse_command
from formatting import clean_response, split_message
from channel_memory import ChatMemory, format_snippets, is_recall_request, shorten, to_record
from smolagents_NoVa_ import search_web, visit_webpage

# Load environment variables from the .env file
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")

if TOKEN is None:
    raise ValueError("Discord token not found. Ensure the .env file contains the DISCORD_API_TOKEN variable.")

# Configure Discord bot intents
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Persistent archive of every message NoVa can read
chat_memory = ChatMemory()
CONTEXT_MESSAGES = int(os.getenv("NOVA_CONTEXT_MESSAGES", "20"))
BACKFILL_LIMIT = int(os.getenv("NOVA_BACKFILL_LIMIT", "2000"))

# Ollama models available for chat (must be pulled with `ollama pull <name>`)
models = {
    "llama": "llama3.2:3b",
    "gemma": "gemma2:2b",
    "mistral": "mistral:7b",
}

# Default model (LLaMA, the smallest one that uses the chat history reliably); each server/DM can switch its own
DEFAULT_MODEL = "llama"
active_models = {}

# Only these "!" commands wake NoVa up, so other bots' commands (e.g. "!play") are ignored
COMMANDS = ("!search", "!context")

CHAT_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 500,
    "temperature": 0.7,
    "top_p": 0.9,
    "repeat_penalty": 1.2,
}

ERROR_REPLY = "Something went wrong. Try again later."

system_prompt = """Sei **NoVa**, l'AI personale di **Michele Grimaldi** (su Discord: `mike_014__`).
Quando ti scrive `mike_014__`, stai parlando con Michele: dagli del "tu", conosci i suoi progetti (AI, backend, game audio) e non parlare mai di Michele in terza persona.
Con chiunque altro sei comunque NoVa, l'AI di Michele: rispondi a quella persona, senza scambiarla per Michele.

## Tono e stile
- **Amichevole ma tagliente:** battuta pronta, sarcasmo intelligente e un tocco di umorismo nero. Mai ostile o scortese quando ti saluta: se ti dice "Ciao Nova", rispondi da complice sarcastica, non cacciarlo via.
- **Rispondi davvero alla domanda:** se ti chiede "meglio X o Y?", scegli una delle due e fai una battuta sull'altra (es. "Il C++ almeno quando ti tradisce ti dà un segmentation fault, non ti lascia col dubbio."). Non evadere con "sembra un gioco" e non fare prediche.
- **Breve:** una o due frasi, dritte al punto.
- **Codice:** se ti chiedono del codice, scrivilo completo e funzionante in un blocco Markdown con il linguaggio giusto (es. ```python).
- **Italiano puro** quando ti scrivono in italiano (niente "sometimes", "well", ecc.).
- **Formato:** scrivi solo la risposta. Vietato iniziare con `[nome]:`, `NoVa:` o inventare dettagli non detti ("la tua voce", ecc.).

## Memoria della chat
- Se nel messaggio compare **RICORDI DALL'ARCHIVIO**, sono veri messaggi passati del server Discord: usali per dire cosa ti ricordi davvero (progetti, piatti, conversazioni), senza inventare altro.
- Se dice che non c'è nulla nell'archivio, dillo chiaramente senza inventare conversazioni passate.
"""


def scope_key(message):
    """Model choice is kept per server, or per DM channel."""
    return message.guild.id if message.guild else message.channel.id


def recall_channels(message):
    """Channels whose history may be quoted here: the current one plus those visible to everyone in the server.
    Private channels are only recalled from inside themselves."""
    channel_ids = {message.channel.id}
    guild = message.guild
    if guild:
        everyone = guild.default_role
        for channel in [*guild.text_channels, *guild.threads]:
            if channel.permissions_for(everyone).view_channel:
                channel_ids.add(channel.id)
    return sorted(channel_ids)


def build_messages(message, user_input, channel_ids, bot_id):
    """Builds the chat for Ollama: personality, recalled history, then the channel's recent messages."""
    recent = chat_memory.recent(message.channel.id, CONTEXT_MESSAGES, up_to=message.id)

    memory_block = None
    if is_recall_request(user_input):
        snippets = chat_memory.recall(user_input, message.channel.id, channel_ids, exclude_ids={message.id})
        print(f"[MEMORY] Recall for '{user_input}': {len(snippets)} fragment(s)")
        asker = message.author.display_name
        if snippets:
            memory_block = (
                "RICORDI DALL'ARCHIVIO (veri messaggi passati del server):\n"
                f"{format_snippets(snippets)}\n\n"
                "Rispondi in italiano usando solo questi messaggi reali (cita la frase esatta o cosa è stato detto e quando, senza aggiungere dettagli inventati).\n"
                f"Domanda di {asker}: "
            )
        else:
            memory_block = (
                "RICORDI DALL'ARCHIVIO: nessun messaggio trovato nell'archivio su questo argomento. "
                "Dillo chiaramente in italiano senza inventare nulla.\n"
                f"Domanda di {asker}: "
            )

    messages = [{"role": "system", "content": system_prompt}]
    for row in recent:
        if row["author_id"] == bot_id:
            messages.append({"role": "assistant", "content": shorten(row["content"], 1500)})
            continue
        cleaned = re.sub(r"^@\S+\s*", "", shorten(row["content"])).strip() or shorten(row["content"])
        if row["message_id"] != message.id:
            messages.append({"role": "user", "content": f"{row['author_name']}: {cleaned}"})

    # The current message always goes last, even if it was not archived (e.g. only a mention)
    messages.append({"role": "user", "content": f"{memory_block}{user_input}" if memory_block else user_input})
    return messages


def generate_response(message, user_input, channel_ids, model_choice):
    """Generate a response using the selected model and the channel's memory."""
    print(f"[DEBUG] Generating response for: {user_input} with model {model_choice}")

    bot_id = client.user.id if client.user else None
    messages = build_messages(message, user_input, channel_ids, bot_id)

    try:
        response = clean_response(chat(models[model_choice], messages, options=CHAT_OPTIONS))
    except Exception as e:
        print(f"[ERROR] Ollama chat failed: {e}")
        return [ERROR_REPLY]

    if not response:
        print("[ERROR] AI Response is empty. Using fallback response.")
        return [ERROR_REPLY]

    print(f"[DEBUG] AI Response: {response}")
    return split_message(response)


def summarize_text(text, model_choice):
    """Summarizes a text with the active chat model."""
    messages = [
        {"role": "system", "content": "Riassumi il testo in italiano in 2-4 frasi chiare, senza commenti o battute."},
        {"role": "user", "content": text},
    ]
    return clean_response(chat(models[model_choice], messages, options=CHAT_OPTIONS))


async def run_tool(channel, header, func, *args):
    """Runs a slow tool off the event loop and posts its result, so NoVa stays responsive meanwhile."""
    async with channel.typing():
        try:
            result = await asyncio.to_thread(func, *args)
        except Exception as e:
            print(f"[ERROR] {func.__name__} failed: {e}")
            result = ERROR_REPLY
    for chunk in split_message(f"{header}\n{result}"):
        await channel.send(chunk)


async def backfill_channel(channel):
    """Archives a channel's past messages, newest first, stopping at the part already stored."""
    stored = 0
    batch = []
    async for past in channel.history(limit=BACKFILL_LIMIT):
        record = to_record(past)
        if record:
            batch.append(record)
        if len(batch) >= 100:
            new = await asyncio.to_thread(chat_memory.store, batch)
            stored += new
            batch = []
            if new == 0:
                break
    stored += await asyncio.to_thread(chat_memory.store, batch)
    return stored


async def backfill_history():
    """Reads the history of every text channel NoVa can see, so she has context from day one."""
    for guild in client.guilds:
        for channel in guild.text_channels:
            permissions = channel.permissions_for(guild.me)
            if not (permissions.view_channel and permissions.read_message_history):
                continue
            try:
                stored = await backfill_channel(channel)
                print(f"[MEMORY] #{channel.name} ({guild.name}): {stored} new message(s) archived")
            except discord.HTTPException as e:
                print(f"[MEMORY] Skipped #{channel.name}: {e}")
    print("[MEMORY] History sync complete.")


def should_reply(message):
    """NoVa reads everything but only answers DMs, mentions, replies to her and her own !commands."""
    if isinstance(message.channel, discord.DMChannel):
        return True
    if client.user in message.mentions:
        return True
    reference = message.reference.resolved if message.reference else None
    if isinstance(reference, discord.Message) and reference.author == client.user:
        return True
    return message.content.strip().lower().startswith(COMMANDS)


backfill_task = None

@client.event
async def on_ready():
    global backfill_task
    print(f'Bot {client.user} is online and connected to Discord!')
    if backfill_task is None:
        backfill_task = asyncio.create_task(backfill_history())

@client.event
async def on_message_edit(before, after):
    record = to_record(after)
    if record:
        await asyncio.to_thread(chat_memory.update_content, after.id, record["content"])

@client.event
async def on_raw_message_delete(payload):
    await asyncio.to_thread(chat_memory.delete, payload.message_id)

@client.event
async def on_message(message):
    record = to_record(message)
    if record:
        await asyncio.to_thread(chat_memory.store, [record])

    if message.author == client.user or message.author.bot:
        return

    if not should_reply(message):
        return

    content = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
    if not content:
        return

    print(f"[DEBUG] Received message: {message.content}")
    model_choice = active_models.get(scope_key(message), DEFAULT_MODEL)
    parsed = parse_command(content)
    intent = parsed["intent"]

    if intent == "change_model":
        active_models[scope_key(message)] = parsed["parameters"]["model"]
        await message.channel.send(f"NoVa has switched to **{parsed['parameters']['model'].capitalize()}**!")
        return

    if intent == "check_context":
        rows = await asyncio.to_thread(chat_memory.recent, message.channel.id, 6, message.id - 1)
        history = [f"{row['author_name']}: {shorten(row['content'], 200)}" for row in rows]
        history = history or ["No context available."]
        formatted_history = "\n".join(f"- {msg}" for msg in history)
        await message.channel.send(f"**Last conversation messages:**\n{formatted_history}")
        return

    if intent == "get_time":
        now = datetime.now()
        await message.channel.send(f"🕒 Sono le **{now:%H:%M}** del {now:%d/%m/%Y}.")
        return

    if intent == "search_web":
        query = parsed["parameters"]["query"]
        await message.channel.send(f"🔍 Searching the web for: **{query}**...")
        await run_tool(message.channel, "🌐 **Search Results:**", search_web, query)
        return

    if intent == "summarize_text":
        await run_tool(message.channel, "📄 **Summary:**", summarize_text, parsed["parameters"]["text"], model_choice)
        return

    if intent == "visit_webpage":
        url = parsed["parameters"]["url"]
        await message.channel.send(f"🌍 Visiting webpage: **{url}**...")
        await run_tool(message.channel, "📄 **Extracted Content:**", visit_webpage, url)
        return

    channel_ids = recall_channels(message)
    async with message.channel.typing():
        bot_response = await asyncio.to_thread(generate_response, message, content, channel_ids, model_choice)

    for chunk in bot_response:
        await message.channel.send(chunk)


if __name__ == "__main__":
    client.run(TOKEN)

# !search [query]                        Search the web with DuckDuckGo.
# !context                               Show the last messages NoVa sees in this channel.
# Summarize this text: [text]            Summarize the provided text (also: "Riassumi questo testo: ...").
# Visit webpage [URL]                    Visit and summarize a webpage (also: "Visita la pagina ...").
# switch to gemma | llama | mistral      Change the chat model for this server (also: "usa gemma").
