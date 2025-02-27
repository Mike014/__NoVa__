from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Initialize the agent with HfApiModel and add the DuckDuckGoSearchTool
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

def search_on_the_web(query: str) -> str:
    """
    Performs a web search using DuckDuckGo via smolagents.

    Args:
        query (str): The query to search for on the web.

    Returns:
        str: The search results as a formatted string.
    """
    try:
        result = agent.run(f"Search for: {query}")
        if isinstance(result, list):
            result = "\n".join(result)  

        return result.strip() if isinstance(result, str) else "No results found."
    
    except Exception as e:
        return f"Error during search: {e}"

# Usage examples
if __name__ == "__main__":
    # Test the web search function
    query = "Best AI frameworks in 2025"
    search_results = search_on_the_web(query)
    print(f"Search results for '{query}':\n{search_results}\n")
