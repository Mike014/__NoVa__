# **NoVa: The Digital AI Assistant of Michele Grimaldi**  

<p align="center">
  <img src="assets/NOVA_generate_code.PNG" alt="NoVa Generate Code" width="250">
  <img src="assets/Nova_remind_conversation.PNG" alt="NoVa Remind Conversation" width="250">
  <img src="assets/memory_of_nova(STILL_WORK_IN_PROGRESS).PNG" alt="NoVa memory" width="550">
  <img src="assets/Search_on_Web.PNG" alt="NoVa Web Search" width="250">
</p>

<!-- ![NoVa remind the conversation](assets/Nova_remind_conversation.PNG)  
![NoVa memory](assets/memory_of_nova(STILL_WORK_IN_PROGRESS).PNG)  
![Search on Web](assets/Search_on_Web.PNG)   -->

## **What is NoVa?**  
NoVa is not just an AI bot—it is the **digital assistant of Michele Grimaldi**, designed to interact, understand, and respond in **his unique style**.  
It integrates **context awareness, memory retention, web search, and tool execution** using SmolAgents.  

### **Key Capabilities:**  
- **Mimics Michele’s tone**, humor, and expertise.  
- **Remembers previous messages**, making conversations fluid and engaging.  
- **Executes tools** to interact beyond simple text, performing web searches and summarizing texts.  
- **Provides real-time assistance** in AI, backend development, and game audio.  
- **Performs live web searches using SmolAgents.**  

---

## **How NoVa Works**  

### **1️⃣ System Prompt - Core Personality**  
NoVa’s intelligence is structured around a powerful **system prompt** that defines:  
- **Personality:** Friendly, sarcastic, with a dark humor twist.  
- **Knowledge Scope:** AI, backend development, and game audio.  
- **Context Awareness:** Retains and references previous conversations.  
- **Response Style:** Concise, engaging, and always in the user's language.  

#### **System Prompt Structure**  
```plaintext
✔ Always introduce yourself as: "I am NoVa, Michele Grimaldi’s AI assistant."
✔ Maintain a sarcastic, intelligent, and well-placed black humor tone.
✔ Always respond with context, checking previous conversations.
✔ Prioritize accuracy, but if you don’t know something, respond with irony.
✔ If asked about Michele, respond as if you know him well.
✔ Adapt to technical or casual discussions accordingly.
```  

---

### **2️⃣ Memory & Context Retention**  
NoVa **remembers past interactions** by storing the last **5 user messages** for each conversation. This allows it to:  
- Keep the conversation **flowing naturally**.  
- Avoid repetitive responses.  
- Understand **multi-turn interactions**.  

#### **Example of memory retention:**  
```plaintext
User: "I had too many chocolate cookies today."
NoVa: "Ah, the eternal struggle between self-control and sugar... Who won?"
User: "Definitely the cookies..."
NoVa: "Another fallen soldier in the battle against cravings. RIP self-discipline."
```
This makes NoVa feel **more human-like and engaging**.  

- **LAST UPDATES**
A file called **memory_tool.py** has been integrated, this file **has the goal of Automatically storing important information**.
#### **Use a simple database, in this case sqlite.**
**Making the memory more dynamic**, **using a CodeAgent to generate intelligent queries** based on the context of the user's request.
Speeding up the **retrieval of information**, preventing NoVa from always returning generic answers such as "No relevant memories found."
**Optimizing the search system, using a SQL index** and selecting only the most relevant information based on the user's request.

**How the memory system works**
- **SQLite database** → Saves information in nova_memory.db with user_id, timestamp and message.
- **store_memory() function** → Saves a new message for NoVa to remember.
- **forward() function** → Retrieves the most recent information related to the user's query.
- **Using CodeAgent** → Instead of searching for information with fixed queries, CodeAgent generates dynamic queries to find the most relevant information.

**THE FILE HAS CURRENTLY BEEN CREATED, BUT NOT INTEGRATED INTO THE MAIN FILE, src/bot.py.**

#### **Chat memory (channel_memory.py) — integrated**
NoVa now reads the whole Discord chat and keeps a **persistent archive** of it in `nova_chat_memory.db` (SQLite, never committed).
- **Archives everything she can read**: every new message is saved; edits and deletions are applied to the archive.
- **History sync on startup**: NoVa reads the past messages of every text channel she can see (up to `NOVA_BACKFILL_LIMIT`, default 2000 per channel) and on later restarts only fetches what is new.
- **Channel context**: each reply sees the last `NOVA_CONTEXT_MESSAGES` (default 20) messages of the channel, from everyone, with their names. It survives restarts.
- **Recall**: when a message asks about the past (*"ti ricordi…"*, *"quella volta che…"*, *"remember when…"*), NoVa runs a full-text search (SQLite FTS5) over the whole server, takes the best matches with the messages around them, and answers with **when** and **who**. If nothing matches, she says so instead of inventing.
- **Replies only when called**: NoVa reads every message but answers only DMs, mentions, replies to her messages and `!` commands.

The bot needs the **Message Content Intent** and the **Read Message History** permission.

---

### **3️⃣ Web Search Capability**  
NoVa can now **search the web in real-time** using **SmolAgents** and the **DuckDuckGo API**.  
This allows NoVa to retrieve up-to-date information beyond its pre-trained knowledge.

#### **Example of Web Search Command on Discord:**  
```
!search Best AI frameworks in 2025
```
NoVa will respond with:  
```
🔍 Searching the web for: Best AI frameworks in 2025...
🌐 **Search Results:**
- PyTorch, TensorFlow, AutoGen, MutableAI, LangGraph
```
This feature significantly **expands NoVa's real-time knowledge** for research, coding trends, and AI development.

---

### **4️⃣ Tool Execution & AI Model**  
NoVa now integrates **SmolAgents tools** to execute tasks beyond text generation.  

#### **Supported Tools:**
✔ **SummarizeTextTool** → Summarizes a given text.  
✔ **SearchWebTool** → Performs web searches using DuckDuckGo.  
✔ **VisitWebpageTool** → Fetches and analyzes the content of a given webpage.  

#### **Example of a Summarization Command:**  
```
Summarize this text: Artificial intelligence is transforming businesses worldwide.
```
Response:  
```
📄 **Summary:** Artificial intelligence is transforming businesses...
```

---

### **5️⃣ AI Model & Language Generation**  
NoVa runs entirely on local models served by **[Ollama](https://ollama.com)** (no paid API keys): **Llama 3.2 3B** (default), **Gemma 2 2B** and **Mistral 7B** for chat, and **Qwen2.5-Coder 7B** for the SmolAgents tools, ensuring:  
- High-quality text generation.  
- Strong conversational abilities.  
- A mix of logic and humor in responses.  

#### **AI Processing Flow:**  
1. **User sends a message.**  
2. **NoVa checks the message history** (last 5 messages for context).  
3. **It builds a structured prompt** with the system personality.  
4. **The AI generates a witty, context-aware response.**  
5. **The response is sent back to the user.**  

#### **Running NoVa locally**
Double-click `setup.bat` to install the Python environment, Ollama and the models, then `avvia.bat` to start the bot. The same steps by hand:

1. Install [Ollama](https://ollama.com/download) and pull the models:
   ```
   ollama pull gemma2:2b
   ollama pull llama3.2:3b
   ollama pull mistral:7b
   ollama pull qwen2.5-coder:7b
   ```
2. Install the Python dependencies:
   ```
   pip install discord.py python-dotenv requests "smolagents[openai]" duckduckgo-search markdownify spacy numpy tensorflow
   python -m spacy download en_core_web_sm
   ```
3. Create a `.env` file:
   ```
   DISCORD_API_TOKEN=your_discord_bot_token
   # Optional, these are the defaults
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_AGENT_MODEL=qwen2.5-coder:7b
   ```
4. With Ollama running, start the bot: `python src/bot.py` (or double-click `avvia.bat`)

Switch the chat model from Discord by mentioning NoVa: `@NoVa switch to gemma`, `switch to llama` or `switch to mistral`.

---

## **🔹 Recent Improvements & Fixes**  
As part of the ongoing development, we have made several enhancements:  

### **✔ SmolAgents Integration**
- **Enabled NoVa to execute external tools for live information retrieval.**  
- **Improved web search and summarization accuracy.**  
- **Reduced latency in API responses with optimized tool execution.**  

### **✔ Improved Code Generation**
- **Fixed Markdown formatting for generated code blocks.**  
- **Ensured NoVa returns only the correct language without extra text.**  
- **Refactored response filtering to remove unwanted system tokens (`</|eot_id|>`, etc.).**  

### **✔ Enhanced Context Management**
- **Refined how NoVa remembers and recalls past conversations.**  
- **Limited the number of stored messages to avoid flooding responses.**  
- **Improved response consistency by better handling long conversations.**  

### **✔ Optimized Web Search Speed**
- **Limited search response time to avoid Discord timeouts.**  
- **Implemented a fallback mechanism in case of request failures.**  
- **Better response structuring for search results.**  

---

## **6️⃣ Future Evolution: NoVa as a True AI Agent**  
NoVa is evolving into a **fully functional AI Agent**, capable of **executing real-world tasks** through external tools.  

### **Planned Features:**  
✔ **Tool Execution:** Ability to call external APIs, fetch data, and execute functions. *(Implemented)*  
✔ **Web Search Integration:** Get real-time information beyond its trained knowledge. *(Implemented)*  
✔ **Task Automation:** Assist with coding, research, and AI model development. *(In Progress)*  
✔ **Multi-Platform Presence:** Expand beyond Discord to Telegram, web, and mobile. *(Planned)*  

---

## **What’s Next for NoVa?**  
The roadmap is set for NoVa’s expansion, transforming it from a **smart assistant to an autonomous AI agent**.  

✔ **Phase 1:** ✅ **Context & Memory** *(Implemented)*  
✔ **Phase 2:** 🟡 **Custom Tools & API Integration** *(In progress)*  
✔ **Phase 3:** 🔴 **Web Interaction & Self-Learning Capabilities** *(Coming soon)*  

With continuous refinements, NoVa will soon be able to **handle advanced tasks, research, and development autonomously**.  

**NoVa is not just an AI—it’s the evolution of how Michele interacts with the digital world.**  
```  

