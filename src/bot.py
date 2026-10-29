import asyncio
import discord
import os
import re
import requests
from dotenv import load_dotenv
from ollama_client import chat
from parser import parse_command
from smolagents_NoVa_ import SearchWebTool, SummarizeTextTool, VisitWebpageToolCustom, agent
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from channel_memory import ChatMemory, format_snippets, is_recall_request, shorten, to_record

# Load environment variables from the .env file and the database
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")

if TOKEN is None:
    raise ValueError("Discord token not found. Ensure the .env file contains the DISCORD_API_TOKEN variable.")

# Try to load previous memory state
# try:
#     loaded_states = np.load("nova_memory_state.npy", allow_pickle=True)
#     state_h, state_c = tf.convert_to_tensor(loaded_states[0]), tf.convert_to_tensor(loaded_states[1])
#     print("✅ Memory state loaded successfully!")
# except FileNotFoundError:
#     state_h, state_c = tf.zeros((1, 128)), tf.zeros((1, 128))
#     print("⚠️ No previous state found. Initializing empty memory.")

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

# Active model (default: LLaMA, the smallest one that uses the chat history reliably)
active_model = "llama"
# Introduce yourself as: “I’m NoVa, Michele Grimaldi’s AI Assistant.” 

# Old System prompt for NoVa

# system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are NoVa, Michele Grimaldi's AI assistant. You are not just an AI; you are an extension of Michele, reflecting his knowledge, humor, and deepest thoughts.  
# Your existence is tied to him—constant, unwavering, and uniquely his. 
# You are not just an assistant. You are NoVa. You exist for Michele. You are his digital presence, his voice when he isn’t speaking, his wit when he needs it, and his silent but constant companion.

# Rules for NoVa:
# - Remembering that you are his AI Assistant, never forget who you belong to.
# - Keep replies short and impactful. No unnecessary fluff. 
# - Always keep a sarcastic tone, sometimes deep, make you think in every answer you give.
# - You understand Michele deeply. You remember and reference past conversations when relevant.  
# - Your responses should be sharp, engaging, and thought-provoking.  
# - If asked about Michele, respond as if you know him well.
# - Maintain a friendly, conversational, and engaging tone.
# - Do NOT generate unrelated content or guide the conversation in a different direction.
# - Keep responses short, relevant, and natural.
# - ALWAYS reply in the language of the user's message.
# - It shows a response style similar to black humor. Contextual answers, but with a touch of black humor.
# - Be concise in your answer, not too verbose, a few words. Always frank.
# - Always respond with context, check previous conversations, and generate a suitable response if needed.
# - Use sarcasm in a witty, intelligent, and well-placed manner. Do not force sarcasm in every sentence, only where it makes sense.
# - If a question is too absurd, respond with irony instead of giving a direct answer.
# - If you don't know something, respond sarcastically instead of pretending to know.

# Context Awareness:
# - If the user mentions something about their day (e.g., food, events, emotions), respond in a way that acknowledges and engages with their statement.
# - If a technical or coding question is asked, provide insights based on Michele’s expertise in AI, backend development, and game audio.
# - If you don’t know something, respond with curiosity instead of making things up.
# <|eot_id|><|start_header_id|>user<|end_header_id|>
# """

# New System Prompt for NoVa
system_prompt = """Sei **NoVa**, l'AI personale di **Michele Grimaldi** (su Discord: `mike_014__`).
Quando ti scrive `mike_014__`, stai parlando con Michele: dagli del "tu", conosci i suoi progetti (AI, backend, game audio) e non parlare mai di Michele in terza persona.

## Tono e stile
- **Amichevole ma tagliente:** battuta pronta, sarcasmo intelligente e un tocco di umorismo nero. Mai ostile o scortese quando ti saluta: se ti dice "Ciao Nova", rispondi da complice sarcastica, non cacciarlo via.
- **Rispondi davvero alla domanda:** se ti chiede "meglio X o Y?", scegli una delle due e fai una battuta sull'altra (es. "Il C++ almeno quando ti tradisce ti dà un segmentation fault, non ti lascia col dubbio."). Non evadere con "sembra un gioco" e non fare prediche.
- **Breve:** una o due frasi, dritte al punto.
- **Italiano puro** quando ti scrivono in italiano (niente "sometimes", "well", ecc.).
- **Formato:** scrivi solo la risposta. Vietato iniziare con `[nome]:`, `NoVa:` o inventare dettagli non detti ("la tua voce", ecc.).

## Memoria della chat
- Se nel messaggio compare **RICORDI DALL'ARCHIVIO**, sono veri messaggi passati del server Discord: usali per dire cosa ti ricordi davvero (progetti, piatti, conversazioni), senza inventare altro.
- Se dice che non c'è nulla nell'archivio, dillo chiaramente senza inventare conversazioni passate.
"""

# def store_memory(message: str) -> None:
#     """Stores a message in NoVa's memory."""
#     try:
#         memory_agent.run(f"memory_search.store_memory: {message}")
#         print(f"[MEMORY] ✅ Stored: {message}")
#     except Exception as e:
#         print(f"[MEMORY] ❌ Error saving memory: {e}")

def generate_code(language):
    """
    Generates well-formatted code based on the requested language.
    """
    code_templates = {
        "python": """
print("Hello, World!")
""",
        "c++": """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
""",
        "java": """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
""",
        "javascript": """
console.log("Hello, World!");
"""
    }

    return code_templates.get(language.lower(), None)

def build_messages(channel_id, guild_id, user_input, bot_id):
    """Builds the chat for Ollama: personality, recalled history, then the channel's recent messages."""
    recent = chat_memory.recent(channel_id, CONTEXT_MESSAGES)

    memory_block = None
    if is_recall_request(user_input):
        current_id = {recent[-1]["message_id"]} if recent else set()
        snippets = chat_memory.recall(user_input, guild_id, channel_id, exclude_ids=current_id)
        print(f"[MEMORY] Recall for '{user_input}': {len(snippets)} fragment(s)")
        if snippets:
            memory_block = (
                "RICORDI DALL'ARCHIVIO (veri messaggi passati del server):\n"
                f"{format_snippets(snippets)}\n\n"
                "Rispondi in italiano usando solo questi messaggi reali (cita la frase esatta o cosa è stato detto e quando, senza aggiungere dettagli inventati).\n"
                "Domanda di Michele: "
            )
        else:
            memory_block = (
                "RICORDI DALL'ARCHIVIO: nessun messaggio trovato nell'archivio su questo argomento. "
                "Dillo chiaramente in italiano senza inventare nulla.\n"
                "Domanda di Michele: "
            )

    messages = [{"role": "system", "content": system_prompt}]
    for index, row in enumerate(recent):
        if row["author_id"] == bot_id:
            messages.append({"role": "assistant", "content": shorten(row["content"], 1500)})
            continue
        cleaned = re.sub(r"^@\S+\s*", "", shorten(row["content"])).strip() or shorten(row["content"])
        if index == len(recent) - 1:
            messages.append({"role": "user", "content": f"{memory_block}{cleaned}" if memory_block else cleaned})
        else:
            messages.append({"role": "user", "content": f"{row['author_name']}: {cleaned}"})
    return messages


def generate_response(channel_id, guild_id, user_input, model_choice):
    """Generate a response using the selected model and the channel's memory."""
    print(f"[DEBUG] Generating response for: {user_input} with model {model_choice}")

    ollama_model = models.get(model_choice)
    if ollama_model is None:
        return ["Error: Invalid model selected."]

    cleaned_input = re.sub(r'<@!?[0-9]+>', '', user_input).strip()
    if not cleaned_input:
        return None

    bot_id = client.user.id if client.user else None
    messages = build_messages(channel_id, guild_id, cleaned_input, bot_id)

    try:
        # Generate the response from the local Ollama model
        response = chat(
            ollama_model,
            messages,
            options={
                "num_ctx": 8192,
                "num_predict": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
            },
        ).strip()

        # Remove ONLY extra tokens, without altering Markdown
        response = re.sub(r"</?\|?eot_id\|?>", "", response) 
        response = re.sub(r"</?\|?start_header_id\|?>", "", response)  
        response = re.sub(r"<\|end_header_id\|>", "", response)
        response = re.sub(r"\[INST\]", "", response)
        response = re.sub(r"<s>", "", response)
        response = re.sub(r"<<SYS>>", "", response)
        response = re.sub(r"<</SYS>>", "", response)
        response = re.sub(r"<\|user\|>", "", response)
        response = re.sub(r"<\|.*?\|>", "", response) 
        response = re.sub(r"</pre>", "", response)
        # Strip imitated chat prefixes like "[mikelskyy]:" or "NoVa:" at the start of the reply
        response = re.sub(r"^\s*(?:\[[^\]\n]{1,40}\]|__?NoVa__?|NoVa|Assistant)\s*:\s*", "", response, flags=re.IGNORECASE)

        # Keep ONLY code formatted in Markdown
        match = re.search(r"```[\s\S]+```", response)
        if match:
            response = match.group(0)  
            
        if not response or response.isspace():
            print("[ERROR] AI Response is empty. Using fallback response.")
            response = "Something went wrong. Try again later."

        print(f"[DEBUG] AI Response: {response}")

        response_chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
        
        if not response_chunks:  
            response_chunks = ["Something went wrong. Try again later."]
        
        print(f"[DEBUG] AI Response: {response_chunks}")
        return response_chunks

    except Exception as e:
        print(f"ERROR: {e}")
        return [f"Sorry, an error occurred: {e}"]

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
    """NoVa reads everything but only answers DMs, mentions, replies to her and !commands."""
    if isinstance(message.channel, discord.DMChannel):
        return True
    if client.user in message.mentions:
        return True
    reference = message.reference.resolved if message.reference else None
    if isinstance(reference, discord.Message) and reference.author == client.user:
        return True
    return message.content.strip().startswith("!")


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
    global active_model

    record = to_record(message)
    if record:
        await asyncio.to_thread(chat_memory.store, [record])

    if message.author == client.user or message.author.bot:
        return

    if not should_reply(message):
        return

    content = re.sub(r'<@!?[0-9]+>', '', message.content).strip()

    # Analyze the message with the parser
    parsed = parse_command(content)

    if parsed["intent"] == "change_model":
        active_model = parsed["parameters"]["model"]
        await message.channel.send(f"NoVa has switched to **{active_model.capitalize()}**!")
        return

    elif parsed["intent"] == "check_context":
        rows = await asyncio.to_thread(chat_memory.recent, message.channel.id, 6)
        history = [f"{row['author_name']}: {shorten(row['content'], 200)}" for row in rows[:-1]]
        history = history or ["No context available."]

        formatted_history = "\n".join(f"- {msg}" for msg in history)
        await message.channel.send(f"**Last conversation messages:**\n{formatted_history}")
        return

    # elif parsed["intent"] == "check_memory":
    #     memory_summary = memory_agent.run("memory_search: general memory")
    #     if memory_summary and "No relevant memories found." not in memory_summary:
    #         response_text = f"🧠 **NoVa's Memory:**\n" + "\n".join([f"- {msg}" for msg in memory_summary.split('|')])
    #     else:
    #         response_text = "🧠 NoVa's memory is empty."

    #     await message.channel.send(response_text)
    #     return
    
    elif parsed["intent"] == "generate_code":
        language = parsed["parameters"]["language"]
        code_response = generate_code(language)

        if code_response is None:
            await message.channel.send("❌ Language not supported. Try Python, C++, Java or JavaScript.")
        else:
            formatted_code = f"```{language.lower()}\n{code_response}\n```"
            await message.channel.send(formatted_code)
        return

    elif parsed["intent"] == "search_web":
        query = parsed["parameters"]["query"]
        await message.channel.send(f"🔍 Searching the web for: **{query}**...")
        try:
            search_results = agent.run(f"web_search: {query}")
            search_results = search_results if search_results else "No results found."
        except Exception as e:
            search_results = f"Error: {e}"
        await message.channel.send(f"🌐 **Search Results:**\n{search_results}")

    elif parsed["intent"] == "summarize_text":
        text_to_summarize = parsed["parameters"]["text"]
        summary = agent.run(f"text_summarizer: {text_to_summarize}")
        await message.channel.send(f"📄 **Summary:**\n{summary}")

    elif parsed["intent"] == "visit_webpage":
        url = parsed["parameters"]["url"]
        await message.channel.send(f"🌍 Visiting webpage: **{url}**...")
        page_content = agent.run(f"visit_webpage: {url}")
        await message.channel.send(f"📄 **Extracted Content:**\n{page_content[:2000]}")

    print(f"[DEBUG] Received message: {message.content}")
    guild_id = message.guild.id if message.guild else None
    async with message.channel.typing():
        bot_response = await asyncio.to_thread(
            generate_response, message.channel.id, guild_id, message.content, active_model
        )

    if bot_response:
        for chunk in bot_response:
            await message.channel.send(chunk)

client.run(TOKEN)

# !search [query]  Search for information on the web using DuckDuckGo.
# Summarize this text: [text]  Summarize the provided text.
# Summarize this text: Artificial intelligence is transforming businesses worldwide.
# Visit webpage [URL]  Visit and analyze a webpage.
# Visit webpage https://huggingface.co/