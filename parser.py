import re
import spacy

# Load the NLP model to analyze the text
nlp = spacy.load("en_core_web_sm")

def parse_command(user_input):
    """
    Analyzes the user's message and returns the intent and relevant parameters.
    """
    user_input = user_input.lower().strip()

    # Recognize the search command
    if user_input.startswith("!search"):
        query = user_input.replace("!search", "").strip()
        return {"intent": "search_web", "parameters": {"query": query}}

    # Commands to change the model
    match_model = re.search(r"(use|switch to|change to)\s*(gemma|llama)", user_input)
    if match_model:
        return {"intent": "change_model", "parameters": {"model": match_model.group(2)}}

    # Commands to check the context (e.g., "What did we say before?")
    match_context = re.search(r"(what did we say|do you remember.*\?)", user_input)
    if match_context:
        return {"intent": "check_context", "parameters": {}}

    # Time recognition
    if "what time is it" in user_input or "tell me the time" in user_input:
        return {"intent": "get_time", "parameters": {}}

    # Recognition of code requests with specific language
    match_code = re.search(r"(write|generate|create).*code (in|for|on) (\w+)", user_input)
    if match_code:
        language = match_code.group(3).lower()
        supported_languages = ["python", "c++", "java", "javascript"]
        if language in supported_languages:
            return {"intent": "generate_code", "parameters": {"language": language}}
        else:
            return {"intent": "unsupported_language", "parameters": {"language": language}}

    return {"intent": "unknown", "parameters": {}}

# Local test of the parser
if __name__ == "__main__":
    test_messages = [
        "!search Best AI frameworks in 2025",
        "Write code in Python",
        "Generate a function to add two numbers in C++",
        "Create Java code for a class",
        "What time is it?",
    ]
    for msg in test_messages:
        print(f"User: {msg} -> Parsed: {parse_command(msg)}")
