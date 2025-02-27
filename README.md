Ecco il **README aggiornato**, includendo la nuova funzionalità di **ricerca sul web con SmolAgents**.  

---

# **NoVa: The Digital Assistant of Michele Grimaldi**  

![NoVa](Nova_pictures.PNG)  
![NoVa generate codes](NOVA_generate_code.PNG)  
![NoVa remind the conversation](Nova_remind_conversation.PNG)  
![Search on Web](Search_on_Web.PNG)  

## **What is NoVa?**  
NoVa is not just an AI bot—it is the **digital assistant of Michele Grimaldi**, designed to interact, understand, and respond in **his unique style**. NoVa is evolving into a true **AI Agent**, integrating **context awareness, memory, web search, and tool execution**.  

### **Key Capabilities:**  
- **Mimics Michele’s tone**, humor, and expertise.  
- **Engages in contextual conversations**, remembering past messages.  
- **Executes tools** (AI functions) to interact beyond simple text.  
- **Provides real-time assistance** in AI, software development, and game audio.  
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

---

### **3️⃣ Web Search Capability**  
NoVa can now **search the web in real-time** using **SmolAgents** and the **DuckDuckGo API**. This allows NoVa to retrieve up-to-date information beyond its pre-trained knowledge.

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

### **4️⃣ AI Model & Language Generation**  
NoVa is powered by **Llama-3.2-3B-Instruct** and **Gemma-2-2B**, ensuring:  
- High-quality text generation.  
- Strong conversational abilities.  
- A mix of logic and humor in responses.  

#### **AI Processing Flow:**  
1. **User sends a message.**  
2. **NoVa checks the message history** (last 5 messages for context).  
3. **It builds a structured prompt** with the system personality.  
4. **The AI generates a witty, context-aware response.**  
5. **The response is sent back to the user.**  

---

## **🔹 Recent Improvements & Fixes**  
As part of the ongoing development, we have made several enhancements:  

### **✔ Improved Code Generation**
- **Fixed Markdown formatting for generated code blocks.**  
- **Ensured NoVa returns only the correct language without extra text.**  
- **Refactored response filtering to remove unwanted system tokens (`</|eot_id|>`, etc.).**  

### **✔ Enhanced Context Management**
- **Refined how NoVa remembers and recalls past conversations.**  
- **Limited the number of stored messages to avoid flooding responses.**  
- **Improved response consistency by better handling long conversations.**  

### **✔ Integrated Real-Time Web Search**
- **Enabled NoVa to search for real-time information via SmolAgents.**  
- **Added intent recognition for `!search` commands.**  
- **Formatted search results for better readability.**  

---

## **5️⃣ Future Evolution: NoVa as a True AI Agent**  
NoVa is being developed into a **fully functional AI Agent**, capable of **executing real-world tasks** through external tools.  

### **Planned Features:**  
✔ **Tool Execution:** Ability to call external APIs, fetch data, and execute functions.  
✔ **Web Search Integration:** Get real-time information beyond its trained knowledge. *(Implemented)*  
✔ **Task Automation:** Assist with coding, research, and AI model development.  
✔ **Multi-Platform Presence:** Expand beyond Discord to Telegram, web, and mobile.  

---

## **What’s Next for NoVa?**  
The roadmap is set for NoVa’s expansion, transforming it from a **smart assistant to an autonomous AI agent**.  

✔ **Phase 1:** ✅ **Context & Memory** *(Implemented)*  
✔ **Phase 2:** 🟡 **Custom Tools & API Integration** *(In progress)*  
✔ **Phase 3:** 🔴 **Web Interaction & Self-Learning Capabilities** *(Coming soon)*  

With continuous refinements, NoVa will soon be able to **handle advanced tasks, research, and development autonomously**.  

**NoVa is not just an AI—it’s the evolution of how Michele interacts with the digital world.** 🚀