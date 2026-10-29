from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool
from ollama_client import agent_model

# One agent with the two web tools; max_steps keeps a confused local model from looping for minutes.
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(max_results=5), VisitWebpageTool(max_output_length=20000)],
    model=agent_model(),
    max_steps=4,
)


def search_web(query: str) -> str:
    """Searches the web and returns a short answer with sources."""
    result = agent.run(
        f"Search the web for: {query}\n"
        "Answer in Italian with at most 5 short bullet points, each with its source link."
    )
    return str(result).strip() or "No results found."


def visit_webpage(url: str) -> str:
    """Reads a webpage and returns a short summary of its content."""
    result = agent.run(
        f"Visit the webpage {url}\n"
        "Summarize its main content in Italian with at most 6 short bullet points."
    )
    return str(result).strip() or "No content retrieved."


if __name__ == "__main__":
    print(search_web("Best AI frameworks in 2025"))
    print(visit_webpage("https://huggingface.co"))
