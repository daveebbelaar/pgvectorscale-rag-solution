from typing import Any, Dict, List, Type

import instructor
from groq import Groq
from pydantic import BaseModel

from config.settings import get_settings


class LLMFactory:
    """
    Factory for creating structured LLM completions using Groq.

    Groq is used for chat/reasoning.
    Embeddings are handled separately using SentenceTransformers.
    """

    def __init__(self, provider: str = "groq"):
        self.provider = provider.lower()

        if self.provider != "groq":
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}. "
                "Currently only 'groq' is configured."
            )

        self.settings = get_settings().groq
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """
        Initialize Groq client wrapped with Instructor
        for structured Pydantic responses.
        """

        if not self.settings.api_key:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Add GROQ_API_KEY to your .env file."
            )

        groq_client = Groq(
            api_key=self.settings.api_key
        )

        return instructor.from_groq(
            groq_client
        )

    def create_completion(
        self,
        response_model: Type[BaseModel],
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Any:
        """
        Create a structured LLM completion using Groq.

        Args:
            response_model:
                Pydantic model defining the expected response structure.

            messages:
                List of chat messages.

            **kwargs:
                Optional overrides:
                - model
                - temperature
                - max_retries
                - max_tokens

        Returns:
            Parsed response matching the provided Pydantic model.
        """

        completion_params = {
            "model": kwargs.get(
                "model",
                self.settings.default_model,
            ),
            "temperature": kwargs.get(
                "temperature",
                self.settings.temperature,
            ),
            "max_retries": kwargs.get(
                "max_retries",
                self.settings.max_retries,
            ),
            "response_model": response_model,
            "messages": messages,
        }

        # Only pass max_tokens when it has a value.
        max_tokens = kwargs.get(
            "max_tokens",
            self.settings.max_tokens,
        )

        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens

        return self.client.chat.completions.create(
            **completion_params
        )
