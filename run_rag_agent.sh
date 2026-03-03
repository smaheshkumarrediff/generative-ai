#!/bin/bash
# Export Pinecone API key (replace with your actual key if not using the example)
export PINECONE_API_KEY=pcsk_5FSViY_Mq4G249mcPEpbnBecjR4uNecCd5Wzr9Jp5hfPP7QtXrKq1Cew9m9fvMznYWfwrg

# Export any additional environment variables required by your Google GenAI setup
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
export GOOGLE_CLOUD_PROJECT=YOUR_GCP_PROJECT_ID
export GOOGLE_CLOUD_LOCATION=global
export MODEL=gemini-3-flash-preview

# Run the RAG agent (adjust the command as needed for your project structure)
python -m rag_agent.agent
