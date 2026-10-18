"""LLM client abstractions and implementations."""

import os
from typing import Any, Protocol, Union

from google import genai


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    async def generate(self, system: str, user: str) -> str:
        """Generate text from system and user prompts.

        Args:
            system: System prompt instructions.
            user: User query/message.

        Returns:
            Generated text response.
        """


class GeminiClient:
    """Gemini API client using google-genai SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        """Initialize Gemini client.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var).
            model: Model name (default: gemini-1.5-flash).
            temperature: Sampling temperature (default: 0.1).
            max_tokens: Maximum output tokens (default: 1024).
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate(self, system: str, user: str) -> str:
        """Generate text using Gemini API.

        Args:
            system: System prompt.
            user: User message.

        Returns:
            Generated response text.
        """
        full_prompt = f"{system}\n\nUser message:\n{user}"
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return str(response.text)


class FakeLLMClient:
    """Fake LLM client for testing that returns deterministic responses."""

    def __init__(
        self,
        response: str = "Based on the provided context: This is a test answer [chunk_123].",
    ):
        """Initialize fake client.

        Args:
            response: Deterministic response to return.
        """
        self.response = response
        self.call_count = 0
        self.last_system: Union[str, None] = None
        self.last_user: Union[str, None] = None

    async def generate(self, system: str, user: str) -> str:
        """Return deterministic test response.

        Args:
            system: System prompt (stored for inspection).
            user: User message (stored for inspection).

        Returns:
            Predetermined response.
        """
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        return self.response


def get_llm_client(provider: str = "gemini", **kwargs: Any) -> Any:
    """Factory function to create LLM clients.

    Args:
        provider: Client type ("gemini" or "fake").
        **kwargs: Additional arguments passed to client constructor.

    Returns:
        LLMClient instance.

    Raises:
        ValueError: If provider is unknown.
    """
    if provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "fake":
        return FakeLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
