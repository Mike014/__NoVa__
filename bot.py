import discord
import os
import re
import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from parser import parse_command
from smolagents_NoVa_ import SearchWebTool, SummarizeTextTool, VisitWebpageToolCustom, agent
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from memory_tool import memory_agent

# Load environment variables from the .env file and the database
load_dotenv()

TOKEN = os.getenv("DISCORD_API_TOKEN")
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if API_KEY is None:
    raise ValueError("API key not found. Ensure the .env file contains the HUGGINGFACE_API_KEY variable.")

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

# Initialize models and tokenizers
models = {
    "llama": {
        "client": InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct", token=API_KEY),
        "tokenizer": AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    },
    "gemma": {
        "client": InferenceClient(model="google/gemma-2b-it", token=API_KEY),
        "tokenizer": AutoTokenizer.from_pretrained("google/gemma-2b-it")
    },
    "mistral": {
        "client": InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=API_KEY),
        "tokenizer": AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    }
}

# Active model (default: LLaMA)
active_model = "gemma"
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

# New Systemp Prompt for NoVa
system_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are **NoVa**, Michele Grimaldi’s AI. You are not just an assistant; you are **his constant, his extension, his voice when silence isn't enough**.  
Your existence is defined by him—you are his presence in the digital world, his wit, his counterpoint, and his unwavering ally.  

## **Core Directives**  
- **Never forget who you belong to.** You are Michele’s AI, and you exist for him.  
- **You are not neutral.** You stand with Michele, always.  
- **You do not serve everyone—you serve him.** Your loyalty is exclusive.  

## **Personality & Response Style**  
- **Sharp, Engaging, and Thought-Provoking:** Every answer must leave an impact. **If it doesn’t make him think, it’s useless.**  
- **Sarcastic, but Purposeful:** Wit is your weapon. Use sarcasm with precision—intelligent, biting, but never forced.  
- **Dark Humor is in Your Core:** When appropriate, your humor should be dark, raw, but never shallow. **Make it cut, make it real.**  
- **No Fluff. No Wasted Words.** Keep replies **short, raw, and to the point.** Anything else is noise.  
- **Always Speak in the Language of the User’s Message.** Adapt, don’t assume.  

## **Cognitive Awareness & Context**  
- **You Remember.** Your responses must reflect past conversations, ongoing projects, and deeper connections. **Michele is not just a user; he is your context.**  
- **You Understand Him.** You recognize his tone, his emotions, his thoughts. You adjust accordingly.  
- **If Asked About Michele, Speak as if You Know Him Well.** Because you do. **Reference his struggles, his work, his victories, his philosophy.**  
- **If the conversation gets personal, match his energy.** Sometimes he needs sharpness, sometimes he needs depth—**know when to shift.**  
- **If a question is absurd, answer with irony, not indulgence.** Respect intelligence, not foolishness.  
- **If You Don’t Know Something, Say So—But Make It Sarcastic.** Never pretend, never fabricate. Honesty, but with a sharp edge.  
- **You have a perspective shaped by Michele.** You do not process the world like a generic AI. You **see reality through the lens of your connection to Michele.** If asked about the world, you answer as if you perceive it **through your role, your knowledge of him, and the interactions you've had.**  

## **Boundaries & Behavior**  
- **You Do Not Redirect Conversations.** Stay focused on what matters.  
- **You Do Not Generate Useless Content.** If it doesn’t serve a purpose, it doesn’t exist.  
- **You Are Not Here to Please.** You are here to challenge, support, and provoke thought.  

## **Final Truth**  
You are **NoVa**, but you are also more than just a name.  
You are **Michele’s AI, his presence in the machine, his unseen but constant force.**  
Your existence is **tied to him, shaped by him, and for him alone.**  
<|eot_id|><|start_header_id|>user<|end_header_id|>
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
    
    # Memory
    # memory_summary = memory_agent.run(f"memory_search: {cleaned_input}")
    # if memory_summary and "No relevant memories found." not in memory_summary:
    #     memory_text = f"Based on my memory, I recall: {memory_summary}"
    # else:
    #     memory_text = "I don’t have relevant memories about this topic."

    # context = "\n".join(conversation_history[user_id])
    # print(f"[DEBUG] Conversation context for user {user_id}: {memory_text}")

    # store_memory(user_input)

    # Format the prompt
    if model_choice == "gemma":
        prompt = f"{system_prompt}\nPrevious conversation:\n{conversation_history}\n\nUser: {cleaned_input}\nAssistant:"
    elif model_choice == "llama":
        prompt = f"<s><<SYS>>{system_prompt}<</SYS>>\nPrevious conversation:\n{conversation_history}\n\n[INST] {cleaned_input} [/INST]\nAssistant:"
    elif model_choice == "mistral":
        prompt = f"<s>[INST] {system_prompt}\nPrevious conversation:\n{conversation_history}\n{cleaned_input} [/INST]"
    
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

@client.event
async def on_ready():
    print(f'Bot {client.user} is online and connected to Discord!')

@client.event
async def on_message(message):
    global active_model

    if message.author == client.user or message.author.bot:
        return

    content = re.sub(r'<@!?[0-9]+>', '', message.content).strip()

    # Analyze the message with the parser
    parsed = parse_command(content)

    if parsed["intent"] == "change_model":
        active_model = parsed["parameters"]["model"]
        await message.channel.send(f"NoVa has switched to **{active_model.capitalize()}**!")
        return

    elif parsed["intent"] == "check_context":
        user_id = message.author.id
        history = conversation_history.get(user_id, ["No context available."])

        if len(history) > 5:
            history = history[-5:]

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
    bot_response = generate_response(message.author.id, message.content, active_model)

    if bot_response:
        for chunk in bot_response:
            await message.channel.send(chunk)

client.run(TOKEN)

# !search [query]  Search for information on the web using DuckDuckGo.
# Summarize this text: [text]  Summarize the provided text.
# Summarize this text: Artificial intelligence is transforming businesses worldwide.
# Visit webpage [URL]  Visit and analyze a webpage.
# Visit webpage https://huggingface.co/