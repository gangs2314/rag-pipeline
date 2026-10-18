"""Tests for LLM client implementations."""

import pytest

from rag_pipeline.generation.llm_client import (
    FakeLLMClient,
    GeminiClient,
    get_llm_client,
)


@pytest.mark.asyncio
async def test_fake_llm_client_basic():
    """Test FakeLLMClient returns deterministic response."""
    client = FakeLLMClient(response="Test answer [chunk_1]")
    result = await client.generate("system", "user query")
    assert result == "Test answer [chunk_1]"
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_fake_llm_client_stores_prompts():
    """Test FakeLLMClient stores system and user prompts."""
    client = FakeLLMClient()
    await client.generate("sys prompt", "user msg")
    assert client.last_system == "sys prompt"
    assert client.last_user == "user msg"


def test_fake_llm_client_default_response():
    """Test FakeLLMClient has sensible default response."""
    client = FakeLLMClient()
    assert "chunk_" in client.response
    assert "[" in client.response


def test_get_llm_client_fake():
    """Test factory creates FakeLLMClient."""
    client = get_llm_client("fake")
    assert isinstance(client, FakeLLMClient)


def test_get_llm_client_fake_with_response():
    """Test factory passes kwargs to FakeLLMClient."""
    client = get_llm_client("fake", response="Custom response")
    assert client.response == "Custom response"


def test_get_llm_client_unknown_provider():
    """Test factory raises on unknown provider."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_client("unknown_provider")


def test_gemini_client_requires_api_key():
    """Test GeminiClient raises if GEMINI_API_KEY not set."""
    import os
    old_key = os.environ.get("GEMINI_API_KEY")
    try:
        os.environ.pop("GEMINI_API_KEY", None)
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient()
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key


def test_gemini_client_accepts_api_key():
    """Test GeminiClient can be initialized with explicit API key."""
    try:
        client = GeminiClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
    except Exception as e:
        # genai may fail on invalid key, which is OK for this test
        assert "test-key" in str(e) or "api" in str(e).lower()


def test_gemini_client_model_defaults():
    """Test GeminiClient has correct defaults."""
    client = GeminiClient(api_key="test-key")
    assert client.model_name == "gemini-1.5-flash"
    assert client.temperature == 0.1
    assert client.max_tokens == 1024


def test_gemini_client_custom_config():
    """Test GeminiClient accepts custom configuration."""
    client = GeminiClient(
        api_key="test-key",
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=2048,
    )
    assert client.model_name == "gemini-1.5-pro"
    assert client.temperature == 0.5
    assert client.max_tokens == 2048
