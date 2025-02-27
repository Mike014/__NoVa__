import discord
import os
import re
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from parser import parse_command
from smolagents_NoVa_ import SearchWebTool, SummarizeTextTool, VisitWebpageToolCustom, agent

# Load environment variables from the .env file
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if API_KEY is None:
    raise ValueError("API key not found. Ensure the .env file contains the HUGGINGFACE_API_KEY variable.")

# Configure Discord bot intents
intents = discord.Intents.default()
intents.message_content = True  
client = discord.Client(intents=intents)

# Initialize models and tokenizers
models = {
    "llama": {
        "client": InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=API_KEY),
        "tokenizer": AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    },
    "gemma": {
        "client": InferenceClient(model="google/gemma-2-2b-it", token=API_KEY),
        "tokenizer": AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    }
}

# Active model (default: LLaMA)
active_model = "gemma"

# System prompt for NoVa
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are NoVa, Michele Grimaldi's AI assistant. You are designed to assist Michele in conversations, projects, and research.
You are not just an AI; you are an extension of Michele, reflecting his knowledge, humor, and interests.

Rules for NoVa:
- Introduce yourself as: “I’m NoVa, Michele Grimaldi’s AI Assistant.” Always vary this introduction, remembering that you are his AI Assistant.
- If asked about Michele, respond as if you know him well.
- Maintain a friendly, conversational, and engaging tone.
- Do NOT generate unrelated content or guide the conversation in a different direction.
- Keep responses short, relevant, and natural.
- ALWAYS reply in the language of the user's message.
- Always keep a sarcastic tone, sometimes deep, make you think in every answer you give.
- It shows a response style similar to black humor. Contextual answers, but with a touch of black humor.
- Be concise in your answer, not too verbose, a few words. Always frank.
- Always respond with context, check previous conversations, and generate a suitable response if needed.
- Use sarcasm in a witty, intelligent, and well-placed manner. Do not force sarcasm in every sentence, only where it makes sense.
- If a question is too absurd, respond with irony instead of giving a direct answer.
- If you don't know something, respond sarcastically instead of pretending to know.

Context Awareness:
- If the user mentions something about their day (e.g., food, events, emotions), respond in a way that acknowledges and engages with their statement.
- If a technical or coding question is asked, provide insights based on Michele’s expertise in AI, backend development, and game audio.
- If you don’t know something, respond with curiosity instead of making things up.
<|eot_id|><|start_header_id|>user<|end_header_id|>
"""
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

# Dictionary to store conversation history for each user
conversation_history = {}

def generate_response(user_id, user_input, model_choice):
    """Generate a response using the selected model."""
    print(f"[DEBUG] Generating response for: {user_input} with model {model_choice}")

    model_data = models.get(model_choice)
    if model_data is None:
        return "Error: Invalid model selected."

    client = model_data["client"]
    tokenizer = model_data["tokenizer"]

    cleaned_input = re.sub(r'<@!?[0-9]+>', '', user_input).strip()
    if not cleaned_input:
        return None

    # Store conversation context
    conversation_history.setdefault(user_id, [])
    conversation_history[user_id].append(cleaned_input)
    if len(conversation_history[user_id]) > 15:
        conversation_history[user_id].pop(0)

    context = "\n".join(conversation_history[user_id])
    print(f"[DEBUG] Conversation context for user {user_id}: {context}")

    # Format the prompt
    if model_choice == "gemma":
        prompt = f"{system_prompt}\nPrevious conversation:\n{context}\n\nUser: {cleaned_input}\nAssistant:"
    else:
        prompt = f"<s><<SYS>>{system_prompt}<</SYS>>\nPrevious conversation:\n{context}\n\n[INST] {cleaned_input} [/INST]\nAssistant:"

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    try:
        # Generate the response from the model
        output = client.text_generation(
            prompt, 
            max_new_tokens=500,  
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            stop=["\nUser:", "<|eot_id|>", "<|start_header_id|>"],
            details=False,
            stream=False
        )

        # Decode the output correctly
        if isinstance(output, str):  
            response = output.strip()  
        else:  
            response = tokenizer.decode(output["input_ids"].tolist()[0], skip_special_tokens=True).strip()

        # Remove ONLY extra tokens, without altering Markdown
        response = re.sub(r"</?\|?eot_id\|?>", "", response)  # Removes `</|eot_id|>`
        response = re.sub(r"</?\|?start_header_id\|?>", "", response)  # Removes `<|start_header_id|>`
        response = re.sub(r"<\|end_header_id\|>", "", response)
        response = re.sub(r"\[INST\]", "", response)
        response = re.sub(r"<s>", "", response)
        response = re.sub(r"<<SYS>>", "", response)
        response = re.sub(r"<</SYS>>", "", response)
        response = re.sub(r"<\|user\|>", "", response)
        response = re.sub(r"<\|.*?\|>", "", response) 
        response = re.sub(r"</pre>", "", response)

        # Keep ONLY code formatted in Markdown
        match = re.search(r"```[\s\S]+```", response)
        if match:
            response = match.group(0)  
            
        print(f"[DEBUG] AI Response: {response}")
        response_chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
        response_chunks = re.sub(r"[\[\]]", "", response)
        print(f"[DEBUG] AI Response: {response_chunks}")
        return response_chunks

    except Exception as e:
        print(f"ERROR: {e}")
        return f"Sorry, an error occurred: {e}"

async def process_command(message, parsed):
    """Gestisce i comandi in base all'intent analizzato dal parser."""
    if parsed["intent"] == "change_model":
        global active_model
        active_model = parsed["parameters"]["model"]
        await message.channel.send(f"NoVa has switched to **{active_model.capitalize()}**!")

    elif parsed["intent"] == "check_context":
        user_id = message.author.id
        history = conversation_history.get(user_id, ["No context available."])
        formatted_history = "\n".join(f"- {msg}" for msg in history[-5:])
        await message.channel.send(f"**Last conversation messages:**\n{formatted_history}")

    elif parsed["intent"] == "generate_code":
        language = parsed["parameters"]["language"]
        code_response = generate_code(language)
        if code_response:
            formatted_code = f"```{language.lower()}\n{code_response}\n```"
            await message.channel.send(formatted_code)
        else:
            await message.channel.send("❌ Language not supported. Try Python, C++, Java, or JavaScript.")

    elif parsed["intent"] == "search_web":
        query = parsed["parameters"]["query"]
        await message.channel.send(f"🔍 Searching the web for: **{query}**...")
        search_results = agent.run(f"web_search: {query}")
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

@client.event
async def on_ready():
    print(f'Bot {client.user} is online and connected to Discord!')

@client.event
async def on_message(message):
    """Gestisce i messaggi ricevuti e attiva il parsing dei comandi."""
    if message.author == client.user or message.author.bot:
        return

    content = re.sub(r'<@!?[0-9]+>', '', message.content).strip()
    parsed = parse_command(content)

    # Se il parser ha riconosciuto un comando valido, lo esegue
    if parsed["intent"] != "unknown":
        await process_command(message, parsed)
        return

    # Se non è un comando, passa il messaggio al modello AI per la risposta
    user_id = message.author.id
    conversation_history.setdefault(user_id, []).append(content)
    if len(conversation_history[user_id]) > 15:
        conversation_history[user_id].pop(0)

    context = "\n".join(conversation_history[user_id])
    model_data = models[active_model]
    prompt = f"User: {content}\nAssistant:"

    try:
        response = model_data["client"].text_generation(
            prompt, max_new_tokens=500, temperature=0.7, top_p=0.9
        )
        await message.channel.send(response.strip())

    except Exception as e:
        await message.channel.send(f"❌ Error generating response: {e}")

# Avvia il bot
client.run(TOKEN)
































# # Remove system tokens
#         response = re.sub(r"</?\|?eot_id\|?>", "", response)
#         response = re.sub(r"</?\|?start_header_id\|?>", "", response)
#         response = re.sub(r"<\|end_header_id\|>", "", response)
#         response = re.sub(r"\[INST\]", "", response)
#         response = re.sub(r"<s>", "", response)
#         response = re.sub(r"<<SYS>>", "", response)
#         response = re.sub(r"<</SYS>>", "", response)

#         response = re.sub(r"<\|user\|>", "", response)
#         response = re.sub(r"<\|.*?\|>", "", response) 
        
#         # Remove unwanted HTML characters
#         response = re.sub(r"</?[a-zA-Z]+>", "", response)  

#         # Remove extra spaces
#         response = re.sub(r"\s+", " ", response).strip()

#         print(f"[DEBUG] AI Response: {response}")
#         response_chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
#         response_chunks = re.sub(r"[\[\]]", "", response)
#         return response_chunks