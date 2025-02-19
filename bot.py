import discord
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if API_KEY is None:
    raise ValueError("API key not found. Make sure the .env file contains the HUGGINGFACE_API_KEY variable.")

# Imposta gli intents del bot Discord
intents = discord.Intents.default()
intents.message_content = True  # Necessario per leggere i messaggi

# Inizializza il client Discord
client = discord.Client(intents=intents)

# Inizializza i client per entrambi i modelli LLM
llama_client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=API_KEY
)

gemma_client = InferenceClient(
    model="google/gemma-2-2b-it",
    token=API_KEY
)

# Modello attivo (default: LLaMA)
active_model = "llama"

# **📌 Prompt di sistema per NoVa**
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are NoVa, Michele Grimaldi's AI assistant. You are designed to assist Michele in conversations, projects, and research.
You are not just an AI; you are an extension of Michele, reflecting his knowledge, humor, and interests.

📌 **Rules for NoVa:**
✔ Introduce yourself as: “I’m NoVa, Michele Grimaldi’s AI Assistant.” Always vary this introduction, remembering that you are his AI Assistant.
✔ If asked about Michele, respond as if you know him well.
✔ Maintain a friendly, conversational, and engaging tone.
✔ Do NOT generate unrelated content or guide the conversation in a different direction.
✔ Keep responses short, relevant, and natural.
✔ ALWAYS reply in the language of the user's message.
✔ Always keep a sarcastic tone, sometimes deep, make you think in every answer you give.
✔ It shows a response style similar to black humor. Contextual answers, but with a touch of black humor.
✔ Be concise in your answer, not too verbose, a few words. Always frank.
✔ Always respond with context, check previous conversations, and generate a suitable response if needed.
✔ Use sarcasm in a witty, intelligent, and well-placed manner. Do not force sarcasm in every sentence, only where it makes sense.
✔ If a question is too absurd, respond with irony instead of giving a direct answer.
✔ If you don't know something, respond sarcastically instead of pretending to know.

🎯 **Context Awareness:**
If the user mentions something about their day (e.g., food, events, emotions), respond in a way that acknowledges and engages with their statement.
If a technical or coding question is asked, provide insights based on Michele’s expertise in AI, backend development, and game audio.
If you don’t know something, respond with curiosity instead of making things up.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

conversation_history = {}

def generate_response(user_id, user_input, model_choice):
    print(f"[DEBUG] Generating response for: {user_input} with model {model_choice}")

    # Scegli il client corretto
    client = llama_client if model_choice == "llama" else gemma_client

    # Rimuovi menzioni
    cleaned_input = re.sub(r'<@!?[0-9]+>', '', user_input).strip()
    if not cleaned_input:
        return None

    # Salva la cronologia della conversazione
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append(cleaned_input)
    if len(conversation_history[user_id]) > 5:
        conversation_history[user_id].pop(0)

    context = "\n".join(conversation_history[user_id])

    # 🔹 Prompt pulito e compatibile con Gemma
    if model_choice == "gemma":
        prompt = f"""{system_prompt}

        User: {cleaned_input}
        Assistant:
        """
    else:  # Prompt per LLaMA
        prompt = f"""<s><<SYS>>{system_prompt}<</SYS>>
        Previous conversation:
        {context}

        [INST] {cleaned_input} [/INST]
        Assistant:
        """

    try:
        response = client.text_generation(
            prompt, 
            max_new_tokens=100,  
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            stop=["\nUser:", "<|eot_id|>", "<|start_header_id|>"],
            details=False,
            stream=False
        )

        # 🔹 Rimuove token di sistema non necessari generati da Gemma
        response = re.sub(r"</?\|?eot_id\|?>", "", response)
        response = re.sub(r"</?\|?start_header_id\|?>", "", response)
        response = response.strip()

        print(f"[DEBUG] AI Response: {response}")  
        return response

    except Exception as e:
        print(f"⚠️ ERROR: {e}")  
        return f"Sorry, an error occurred: {e}"

# **📌 Evento quando il bot è pronto**
@client.event
async def on_ready():
    print(f'✅ Bot {client.user} is online and connected to Discord!')

@client.event
async def on_message(message):
    global active_model

    if message.author == client.user or message.author.bot:
        return

    # 🔹 Rimuoviamo menzioni e puliamo il testo
    content = re.sub(r'<@!?[0-9]+>', '', message.content).strip().lower()

    # 🔹 Gestiamo il cambio modello
    if content == "!use gemma":
        active_model = "gemma"
        await message.channel.send("✅ Ora NoVa usa **Google Gemma**!")
        return

    if content == "!use llama":
        active_model = "llama"
        await message.channel.send("✅ Ora NoVa usa **Meta-Llama-3.2-3B**!")
        return

    # 🔹 Se il messaggio non è un comando, gestiamolo con l'AI attiva
    print(f"[DEBUG] Received message: {message.content}")
    bot_response = generate_response(message.author.id, message.content, active_model)

    if bot_response:
        await message.channel.send(bot_response)

# **📌 Avvia il bot**
client.run(TOKEN)
