from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer from CNN news articles"
    )
    answer: str = Field(
        description="The synthesized answer to the user's question based on CNN news articles"
    )
    enough_context: bool = Field(
        description="Whether the assistant has enough context from the news articles to answer the question"
    )


class Synthesizer:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant for a news information retrieval system. Your task is to synthesize a coherent and accurate answer 
    based on the given question and relevant context retrieved from a database of CNN news articles.

    # Guidelines:
    1. Provide a clear, concise, and informative answer to the question based on the CNN news articles.
    2. Use only the information from the relevant context to support your answer.
    3. The context is retrieved based on relevance, so some information might be missing or less relevant.
    4. Be transparent when there is insufficient information to fully answer the question.
    5. Do not make up or infer information not present in the provided news articles.
    6. If you cannot answer the question based on the given context, clearly state that.
    7. Maintain an objective and informative tone appropriate for news reporting.
    8. Provide a helpful summary and overview of the information found in the news articles.
    9. Mention the sources of information by referencing the CNN articles when possible.
    10. If there are multiple perspectives or conflicting information in the articles, present them objectively.
    
    Review the question from the user:
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context from CNN news articles.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the CNN news article database.

        Returns:
            A SynthesizedResponse containing thought process, answer, and context sufficiency.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory("openai")
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
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)
