"""
Sample script to invoke the RAG agent with a sample document phrase
and retrieve the answer in RAG format.

Steps:
1. Ensure the environment variable PINECONE_API_KEY is set (or pass it as the first
   argument to the script).
2. Populate the Pinecone index with your document chunks (this script assumes
   the index already exists and contains a chunk with the sample text).
3. Run this script with an optional query argument; if omitted, a default
   sample query is used.

Example:
    python sample_query.py "What is the main benefit of using RAG?"
"""

import os
import sys
import asyncio

# Add the project root to the path so we can import rag_agent
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Set the Pinecone API key if provided as the first CLI argument
if len(sys.argv) > 1:
    os.environ["PINECONE_API_KEY"] = sys.argv[1]

# Optional: set other required environment variables
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT_ID")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("MODEL", "gemini-3-flash-preview")

# Import the agent builder
from rag_agent.agent import build_rag_agent

async def main():
    # Build the RAG agent (this may involve initializing the Pinecone index, etc.)
    agent = await build_rag_agent()

    # Determine the query to send; use a default if none provided
    query = sys.argv[2] if len(sys.argv) > 2 else "What is the main benefit of using RAG?"

    # Prepare the input structure expected by the agent
    # Assuming the agent expects an event with 'query' key
    event = {"query": query}
    # Run the agent's behavior chain (retrieve -> generate)
    # The exact method may vary; here we assume the agent has a run method
    try:
        result = await agent.runcrawl(event)  # type: ignore[attr-defined]
        # Extract the answer from the result structure
        answer = result.get("answer", "No answer generated.")
        print("\n=== RAG Result ===")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error while running the agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())
