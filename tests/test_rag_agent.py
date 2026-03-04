import asyncio
import pytest
from unittest.mock import AsyncMock, patch

# Import the classes we want to test
from rag_agent.agent import (
    RetrieveAction,
    GenerateAnswerAction,
    PineconeRetriever,
    build_rag_agent,
)


# ----------------------------------------------------------------------
# Mock data
# ----------------------------------------------------------------------
MOCK_CONTEXT = "This is a mock retrieved context.\nIt contains sample text."
MOCK_ANSWER = "The answer based on the context is: ..."


# ----------------------------------------------------------------------
# Unit tests
# ----------------------------------------------------------------------
class TestPineconeRetriever:
    @patch("rag_agent.agent.SentenceTransformer")
    @patch("rag_agent.agent.pinecone")
    async def test_aretrieve_returns_context(self, mock_pinecone, mock_sentence_transformer):
        # Arrange
        mock_model = AsyncMock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]  # dummy embedding
        mock_sentence_transformer.return_value = mock_model

        # Mock Pinecone index query result
        mock_match = {
            "metadata": {"text": "Sample chunk 1"},
            "id": "1",
            "score": 0.95,
        }
        mock_query_result = {
            "matches": [mock_match],
            "namespace": "test",
        }
        mock_index = AsyncMock()
        mock_index.query.return_value = mock_query_result
        mock_pinecone.Index.return_value = mock_index

        retriever = PineconeRetriever(index_name="test_index")

        # Act
        result = await retriever.aretrieve("sample query")

        # Assert
        self.assertIn("Sample chunk 1", result)
        mock_sentence_transformer.assert_called_once()
        mock_index.query.assert_awaited_once()


class TestRetrieveAction:
    async def test_run_returns_structured_context(self):
        # Arrange
        mock_retriever = AsyncMock()
        mock_retriever.aretrieve = AsyncMock(return_value=MOCK_CONTEXT)
        retrieve_action = RetrieveAction(mock_retriever)

        event = {"input": {"query": "test query"}}
        state = {}

        # Act
        result_state = await retrieve_action.run(event, state)

        # Assert
        assert result_state == {"context": MOCK_CONTEXT}


class TestGenerateAnswerAction:
    async def test_run_returns_answer(self):
        # Arrange
        mock_llm = AsyncMock()
        mock_llm.apredict = AsyncMock(return_value=MOCK_ANSWER)
        generate_action = GenerateAnswerAction(mock_llm)

        event = {
            "input": {
                "context": MOCK_CONTEXT,
                "query": "What is the main benefit?",
            }
        }
        state = {}

        # Act
        result_state = await generate_action.run(event, state)

        # Assert
        assert result_state == {"answer": MOCK_ANSWER}
        mock_llm.apredict.assert_awaited_once()


# ----------------------------------------------------------------------
# Integration‑style test for build_rag_agent (lightweight)
# ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_build_rag_agent_creates_agent_instance():
    """
    Verify that build_rag_agent returns an Agent instance without raising.
    External dependencies are mocked to keep the test fast and deterministic.
    """
    with patch("rag_agent.agent.PineconeRetriever"), patch("rag_agent.agent.AsyncMock") as mock_llm_async:
        # Mock the retriever and LLM instances
        mock_retriever = AsyncMock()
        mock_retriever.aretrieve = AsyncMock(return_value=MOCK_CONTEXT)
        with patch("rag_agent.agent.PineconeRetriever", return_value=mock_retriever):
            with patch("rag_agent.agent.AsyncMock") as mock_llm_class:
                mock_llm_instance = AsyncMock()
                mock_llm_class.return_value = mock_llm_instance
                with patch("rag_agent.agent.AsyncMock", return_value=mock_llm_instance):
                    # Act
                    agent = await build_rag_agent()

                    # Assert
                    assert hasattr(agent, "name")
                    assert agent.name == "RAGAgent"
                    # Ensure the behaviors dict was populated
                    config = agent.runtime_config
                    assert "retrieve" in config.behaviors
                    assert "generate" in config.behaviors
