from smolagents import Tool, CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, HfApiModel

class SummarizeTextTool(Tool):
    """
    This tool summarizes a given text by returning the first half of its words.
    """
    name = "text_summarizer"
    description = "Summarizes a text by providing the first half of its words."

    inputs = {
        "text": {
            "type": "string",
            "description": "The text to be summarized. It should contain at least one sentence.",
        }
    }
    
    output_type = "string"

    def forward(self, text: str) -> str:
        words = text.split()
        return " ".join(words[:len(words) // 2]) if words else "No content to summarize."


class SearchWebTool(Tool):
    """
    This tool performs a web search using DuckDuckGo.
    """
    name = "web_search"
    description = "Performs an online search and returns the best results."

    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to execute on DuckDuckGo.",
        }
    }
    
    output_type = "string"

    def forward(self, query: str) -> str:
        agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())
        result = agent.run(query)
        return "\n".join(result) if isinstance(result, list) else result.strip()


class VisitWebpageToolCustom(Tool):
    """
    This tool visits a webpage and analyzes its content.
    """
    name = "visit_webpage"
    description = "Fetches and analyzes the content of a given webpage."

    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to visit.",
        }
    }
    
    output_type = "string"

    def forward(self, url: str) -> str:
        agent = CodeAgent(tools=[VisitWebpageTool()], model=HfApiModel())
        result = agent.run(url)
        return result if isinstance(result, str) else "No content retrieved."


# Instantiate the tools
summarize_tool = SummarizeTextTool()
search_tool = SearchWebTool()
visit_webpage_tool = VisitWebpageToolCustom()

model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")


# Create an agent with all tools
agent = CodeAgent(tools=[summarize_tool, search_tool, visit_webpage_tool], model=model)

def main():
    # # Example of a web search
    # query = "Best AI frameworks in 2025"
    # search_results = agent.run("web_search: " + query)
    # print(f"Search results for '{query}':\n{search_results}\n")

    # # Example of text summarization
    # sample_text = (
    #     "Artificial intelligence is transforming the world by automating tasks and "
    #     "providing insights across various industries. With continuous innovation "
    #     "and integration into daily life, AI systems are becoming more capable and accessible."
    # )
    # summary = agent.run("text_summarizer: " + sample_text)
    # print("Original text:\n", sample_text)
    # print("\nSummary (first half of the words):\n", summary)

    # Example of visiting a webpage
    url = "https://www.linkedin.com/in/michele-grimaldi-599b36280/"
    webpage_content = agent.run("visit_webpage: " + url)
    print(f"\nContent from '{url}':\n{webpage_content}")

if __name__ == "__main__":
    main()