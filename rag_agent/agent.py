import os
from google.adk.agent import Agent
from google.adk.runtime import Runtime
from google.adk.runtime.config import RuntimeConfig
from google.adk.runtime.behavior import Behavior
from google.adk.runtime.behavior import Action
from google.adk.runtime.events import Event
from google.adk.runtime.types import Struct
from typing import List, Dict, Optional

"""
Google ADK RAG Agent
====================

This module defines a simple Retrieval‑Augmented Generation (RAG) agent
that uses Pinecone for vector storage and a language model for answer
generation.  It also contains helper utilities for working with the
Schwab API, including obtaining an **access token** and a **refresh token**.

Obtaining Schwab Access & Refresh Tokens
----------------------------------------

Schwab's API uses OAuth 2.0.  The typical flow is:

1. **Register an application** on the Schwab Developer Portal to obtain:
   - ``client_id`` (also called ``apiKey``)
   - ``client_secret``

2. **Request an authorization code**:
   - Direct the user (or yourself in a test script) to the
     authorization endpoint:
     ``https://api.schwab.com/oauth/authorize``
   - Include the ``response_type=code``, ``client_id``, ``redirect_uri``,
     and the ``scope`` you need (e.g., ``accounts read``).
   - After the user consents, Schwab redirects to the ``redirect_uri``
     with a ``code`` query parameter.

3. **Exchange the code for tokens**:
   - POST to ``https://api.schwab.com/oauth/token`` with:
     - ``grant_type=authorization_code``
     - ``code`` (the code from step 2)
     - ``redirect_uri`` (must match the one used in step 2)
     - ``client_id`` and ``client_secret`` (Basic Auth header or form data)
   - The response contains:
     - ``access_token`` – short‑lived (typically 1 hour)
     - ``refresh_token`` – long‑lived token used to obtain new access tokens
     - ``expires_in`` – seconds until expiration
     - ``token_type`` (usually ``Bearer``)

4. **Refresh the access token** when it expires:
   - POST to ``https://api.schwab.com/oauth/token`` again, this time with:
     - ``grant_type=refresh_token``
     - ``refresh_token`` (the refresh token you stored)
     - ``client_id`` and ``client_secret``
   - The response gives a new ``access_token`` (and optionally a new
     ``refresh_token``).

**Storing Tokens**

- For local development you can keep the tokens in environment variables:
  ``SCHWAB_ACCESS_TOKEN`` and ``SCHWAB_REFRESH_TOKEN``.
- In production you should store them securely (e.g., secret manager,
  encrypted database) and rotate them regularly.

**Example (pseudo‑code)**

.. code-block:: python

    import requests
    import base64
    import os

    CLIENT_ID = os.getenv("SCHWAB_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SCHWAB_CLIENT_SECRET")
    REDIRECT_URI = os.getenv("SCHWAB_REDIRECT_URI")

    # 1️⃣ Get authorization code (user interaction required)
    auth_url = (
        "https://api.schwab.com/oauth/authorize"
        f"?response_type=code&client_id={CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}&scope=accounts%20read"
    )
    # ... open browser, user logs in, gets redirected with ?code=...

    # 2️⃣ Exchange code for tokens
    token_url = "https://api.schwab.com/oauth/token"
    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    token_payload = {
        "grant_type": "authorization_code",
        "code": AUTHORIZATION_CODE,
        "redirect_uri": REDIRECT_URI,
    }
    headers = {"Authorization": f"Basic {auth_header}"}
    resp = requests.post(token_url, data=token_payload, headers=headers)
    tokens = resp.json()
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    # 3️⃣ Use the access token in subsequent API calls
    #    When it expires, repeat step 2 with grant_type=refresh_token.
"""

# Simple RAG behavior for retrieval
class RetrieveAction(Action):
    def __init__(self, retriever):
        self.retriever = retriever

    async def run(self, event: Event, state: Struct) -> Struct:
        query = event.input.get("query")
        # Retrieve relevant chunks using embeddings stored in Pinecone
        context = await self.retriever.aretrieve(query)
        return Struct({"context": context})

# Simple RAG behavior for answer generation
class GenerateAnswerAction(Action):
    def __init__(self, llm):
        self.llm = llm

    async def run(self, event: Event, state: Struct) -> Struct:
        context = event.input.get("context")
        query = event.input.get("query")
        prompt = (
            f"Use the following context to answer the question:\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        # Placeholder for actual LLM call
        response = await self.llm.apredict(prompt)
        return Struct({"answer": response})

# Retriever that uses Pinecone to store and query document chunks
class PineconeRetriever:
    def __init__(self, index_name: str, embedding_model="text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize the PineconeRetriever.

        Args:
            index_name: Name of the Pinecone index that stores document chunks.
            embedding_model: The embedding model to use (default: "text-embedding-3-small").
            api_key: Optional API key. If not provided, the key is read from the
                     PINECONE_API_KEY environment variable.

        Raises:
            ValueError: If no API key is found in the environment and none is passed.
        """
        self.index_name = index_name
        self.embedding_model = embedding_model
        # Initialize Pinecone client
        import pinecone
        # The Pinecone API key can be supplied either via the `api_key` argument
        # or through the PINECONE_API_KEY environment variable.
        # Example: export PINECONE_API_KEY="your_api_key_here"
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable not set and no api_key provided. "
                "Set it via the environment or pass it explicitly to PineconeRetriever."
            )
        pinecone.init(api_key=api_key, environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp"))
        self.index = pinecone.Index(index_name)

    async def aretrieve(self, query: str) -> str:
        # Generate embedding for the query
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.embedding_model)
        query_embedding = model.encode([query])[0].tolist()

        # Query Pinecone
        result = self.index.query(
            vector=query_embedding,
            top_k=5,
            include_values=False,
        )
        # Extract text chunks from matches
        context_chunks = [match["metadata"].get("text", "") for match in result["matches"]]
        # Join chunks into a single string
        return "\n\n---\n\n".join(context_chunks)

# Utility to list the supported ADK run modes
def list_adk_modes() -> List[str]:
    """
    Return a list of supported ADK execution modes.
    Common modes include:
      - "local": Run the agent locally for development and testing.
      - "cloud": Deploy and run the agent on Google Cloud (e.g., Cloud Run, GKE).
      - "test": Execute unit/integration tests that may spin up temporary environments.
    """
    return ["local", "cloud", "test"]

async def build_rag_agent():
    # TODO: Replace with real implementations of retriever and LLM
    retriever = PineconeRetriever(index_name="document-chunks")  # ensure index exists
    llm = ...        # e.g., Gemini, PaLM, or another language model client

    retrieve_behavior = RetrieveAction(retriever)
    generate_behavior = GenerateAnswerAction(llm)

    config = RuntimeConfig(
        root="rag_agent",
        behaviors={
            "retrieve": retrieve_behavior,
            "generate": generate_behavior,
        },
        # Define the execution flow: retrieve -> generate
        next_events={"retrieve": "generate"},
    )

    agent = Agent(
        name="RAGAgent",
        runtime_config=config,
    )
    return agent

if __name__ == "__main__":
    import asyncio
    # Print supported ADK run modes when the module is executed directly
    print("Supported ADK run modes:", list_adk_modes())
    asyncio.run(build_rag_agent())
