import sqlite3
import os
import logging
from smolagents import Tool, CodeAgent, HfApiModel
from huggingface_hub import InferenceClient

# Configuration
API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DB_FILE = "nova_memory.db"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Hugging Face model
llm_client = HfApiModel(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=API_KEY
)

class MemorySearchTool(Tool):
    """Tool for managing NoVa's memory."""
    
    name = "memory_search"
    description = "Searches NoVa's memories based on user query."

    inputs = {
        "query": {
            "type": "string",
            "description": "User query to search through stored memories."
        }
    }
    output_type = "string"

    @staticmethod
    def initialize_db():
        """Initializes the database if it doesn't exist and adds an index to improve speed."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                message TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_message ON memory (message)")
        conn.commit()
        conn.close()
        logging.info("Memory database initialized successfully.")

    @staticmethod
    def store_memory(user_id: int, message: str) -> None:
        """Stores a new message in memory."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO memory (user_id, message) VALUES (?, ?)
            """, (user_id, message))
            conn.commit()
            logging.info(f"Stored: {message}")
        except Exception as e:
            logging.error(f"Error saving memory: {e}")
        finally:
            conn.close()

    def forward(self, query: str) -> str:
        """Performs a memory search based on the query."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT message 
                FROM memory 
                WHERE message LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, ('%' + query + '%',))

            row = cursor.fetchone()

            if row:
                return row[0]  # Return only the first result to improve speed
            return "No relevant memories found."
        except Exception as e:
            logging.error(f"Error searching memory: {e}")
            return "Memory search error."
        finally:
            conn.close()


# Initialize the database before starting NoVa
MemorySearchTool.initialize_db()

# Initialize the memory agent
memory_tool = MemorySearchTool()
memory_agent = CodeAgent(
    model=llm_client,
    tools=[memory_tool],
    name="NoVaMemoryAgent"
)

# ------------------------------------------
# ✅ AUTOMATIC TESTS
# ------------------------------------------
def run_tests():
    logging.info("🔍 Starting tests on memory_tool.py")

    # Test writing to memory
    test_message = "Today I built a memory system."
    memory_tool.store_memory(user_id=1193921368371240982, message=test_message)

    # Test reading from memory
    retrieved_message = memory_tool.forward("memory system")

    if retrieved_message == test_message:
        logging.info("✅ Test passed: the message was stored and retrieved correctly.")
    else:
        logging.error("❌ Test failed: the message was not retrieved correctly.")

    # Test query that should not find anything
    not_found = memory_tool.forward("this query does not exist")
    if not_found == "No relevant memories found.":
        logging.info("✅ Test passed: searching for a non-existent query returned the correct message.")
    else:
        logging.error("❌ Test failed: the non-existent query returned an unexpected value.")

# Run the tests
if __name__ == "__main__":
    run_tests()