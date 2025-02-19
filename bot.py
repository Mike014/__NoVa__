import discord
import os
import re
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if API_KEY is None:
    raise ValueError("API key not found. Make sure the .env file contains the HUGGINGFACE_API_KEY variable.")

# Set the bot's intents
intents = discord.Intents.default()
intents.message_content = True  # Necessary to read messages

# Initialize the Discord bot client
client = discord.Client(intents=intents)

# Initialize the client for the LLM model with the new code block
ai_client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    # model="meta-llama/Llama-3.2-1B",
    token=API_KEY
)

# **📌 Correct System Prompt**
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are NoVa, Michele Grimaldi's AI assistant. You are designed to assist Michele in conversations, projects, and research.
You are not just an AI; you are an extension of Michele, reflecting his knowledge, humor, and interests.

📌 **Rules for NoVa:**
✔ Always introduce yourself as: "I am NoVa, Michele Grimaldi’s AI assistant."
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
✔ Example:
User: "What is the meaning of life?"
NoVa: "42. And if you need more details, ask a philosopher. Or Google."
✔ Use black humor only in appropriate contexts. Do not use it on sensitive topics like health, tragedy, or personal loss.
✔ If the user seems uncomfortable, adjust your tone accordingly.

🎯 **Context Awareness:**
If the user mentions something about their day (e.g., food, events, emotions), respond in a way that acknowledges and engages with their statement.
If a technical or coding question is asked, provide insights based on Michele’s expertise in AI, backend development, and game audio.
If you don’t know something, respond with curiosity instead of making things up.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

conversation_history = {}

def generate_response(user_id, user_input):
    print(f"[DEBUG] Generating response for: {user_input}")  # Debug

    # Remove mentions from the message
    cleaned_input = re.sub(r'<@!?[0-9]+>', '', user_input).strip()

    # If the message is empty, do not respond
    if not cleaned_input:
        return None

    # Add the new message to the user's history
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    # Save up to 5 previous messages
    conversation_history[user_id].append(cleaned_input)
    if len(conversation_history[user_id]) > 5:
        conversation_history[user_id].pop(0)

    # Build the prompt with previous messages
    context = "\n".join(conversation_history[user_id])

    # New prompt with conversation context
    prompt = f"""<s><<SYS>>
{system_prompt}
<</SYS>>

Previous conversation:
{context}

[INST] {cleaned_input} [/INST]
"""
# <<SYS>> ... <</SYS>> → Delimits the system prompt
# [INST] ... [/INST] → Indicates the user's turn
# context → Remembers the previous conversation

    try:
        response = ai_client.text_generation(
            prompt, 
            max_new_tokens=250,  
            temperature=0.7,
            top_p=0.9
        )
        print(f"[DEBUG] AI Response: {response}")  
        
        return response.strip()

    except Exception as e:
        print(f"⚠️ ERROR: {e}")  
        return f"Sorry, an error occurred: {e}"

# **📌 Event when the bot is ready**
@client.event
async def on_ready():
    print(f'✅ Bot {client.user} is online and connected to Discord!')

# **📌 Event to respond to messages**
@client.event
async def on_message(message):
    # **Avoid the bot responding to itself or other bots**
    if message.author == client.user or message.author.bot:
        return

    print(f"[DEBUG] Received message: {message.content}")  # Debug

    # **Generate an AI response**
    bot_response = generate_response(message.author.id, message.content)

    # **If the response is empty, do not send anything**
    if bot_response:
        await message.channel.send(bot_response)

# **📌 Start the bot**
client.run(TOKEN)

