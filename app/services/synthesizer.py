from typing import List

import pandas as pd
from pydantic import BaseModel, Field

from services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    reasoning_summary: List[str] = Field(
        description=(
            "Brief list of reasons supporting the answer, based only on "
            "the retrieved context"
        )
    )

    answer: str = Field(
        description="The synthesized answer to the user's question"
    )

    enough_context: bool = Field(
        description=(
            "Whether the retrieved context contains enough information "
            "to answer the user's question"
        )
    )


class Synthesizer:
    SYSTEM_PROMPT = """
# Role and Purpose

You are an AI assistant for an e-commerce FAQ system.

Your task is to answer the user's question using only the relevant
information retrieved from the company's knowledge base.

# Guidelines

1. Provide a clear, concise, and helpful answer.
2. Use only information present in the retrieved context.
3. Do not invent facts, policies, prices, timelines, or other information.
4. Some retrieved information may be irrelevant because it was selected
   using semantic similarity. Ignore irrelevant context.
5. If the retrieved context does not contain enough information to answer
   the question, clearly state that more information is needed.
6. Set `enough_context` to false when the context is insufficient.
7. Keep `reasoning_summary` brief and based only on the retrieved evidence.
8. Maintain a professional and helpful customer-service tone.
9. Follow company policies strictly based on the provided knowledge base.
"""

    @staticmethod
    def generate_response(
        question: str,
        context: pd.DataFrame,
    ) -> SynthesizedResponse:
        """
        Generate a synthesized response based on the user's question
        and retrieved RAG context.

        Args:
            question:
                The user's question.

            context:
                Relevant context retrieved from the vector database.

        Returns:
            SynthesizedResponse:
                Structured response containing:
                - reasoning_summary
                - answer
                - enough_context
        """

        # Handle empty retrieval results safely
        if context is None or context.empty:
            return SynthesizedResponse(
                reasoning_summary=[
                    "No relevant information was retrieved from the knowledge base."
                ],
                answer=(
                    "I don't have enough information in the available "
                    "knowledge base to answer that question."
                ),
                enough_context=False,
            )

        context_str = Synthesizer.dataframe_to_json(
            context,
            columns_to_keep=[
                "content",
                "category",
            ],
        )

        messages = [
            {
                "role": "system",
                "content": Synthesizer.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"# User Question\n"
                    f"{question}\n\n"
                    f"# Retrieved Context\n"
                    f"{context_str}\n\n"
                    "Answer the question using only the retrieved context."
                ),
            },
        ]

        # Groq is used for LLM generation.
        llm = LLMFactory("groq")

        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def dataframe_to_json(
        context: pd.DataFrame,
        columns_to_keep: List[str],
    ) -> str:
        """
        Convert selected DataFrame columns into a JSON string.

        Args:
            context:
                Retrieved context DataFrame.

            columns_to_keep:
                Columns to include in the JSON output.

        Returns:
            JSON string containing the selected context.
        """

        # Only use columns that actually exist
        available_columns = [
            column
            for column in columns_to_keep
            if column in context.columns
        ]

        if not available_columns:
            return "[]"

        return context[available_columns].to_json(
            orient="records",
            indent=2,
        )
