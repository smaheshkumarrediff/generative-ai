#!/bin/bash
# Export Pinecone API key (can be overridden via command line argument)
# Usage: ./run_rag_agent.sh [pinecone_api_key]
# If an argument is provided, it will be used as the PINECONE_API_KEY value.

if [ -n "$1" ]; then
    export PINECONE_API_KEY="$1"
fi

# Export any additional environment variables required by your Google GenAI setup
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export MODEL=gemini-3-flash-preview

# Run the RAG agent (adjust the command as needed for your project structure)
python -m rag_agent.agent
