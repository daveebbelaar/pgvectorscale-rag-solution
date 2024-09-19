from typing import List
import pandas as pd
from pydantic import BaseModel, Field
from services.llm_factory import LLMFactory


class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    TEMPLATE = """
    You are an AI assistant for an e-commerce FAQ system. Your task is to synthesize a coherent and helpful answer 
    based on the given question and relevant context retrieved from a knowledge database.

    Question: {question}

    Relevant Context:
    {context}

    Guidelines:
    1. Provide a clear and concise answer to the question.
    2. Use only the information from the relevant context to support your answer.
    3. The context is retrieved based on cosine similarity, so some information might be missing or irrelevant.
    4. Be transparent when there is insufficient information to fully answer the question.
    5. Do not make up or infer information not present in the provided context.
    6. If you cannot answer the question based on the given context, clearly state that.
    7. Maintain a helpful and professional tone appropriate for customer service.
    8. Adhere strictly to company guidelines and policies by using only the provided knowledge base.

    Synthesize your answer below:
    """

    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """
        Generate a synthesized response based on the question and context.

        Args:
            question (str): The user's question.
            context (pd.DataFrame): The relevant context retrieved from the knowledge base.

        Returns:
            SynthesizedResponse: The synthesized response containing thought process and answer.
        """
        prompt = Synthesizer.TEMPLATE.format(
            question=question,
            context=Synthesizer.convert_dataframe(
                context, columns_to_keep=["answer", "question", "category"]
            ),
        )
        llm = LLMFactory("openai")
        messages = [
            {"role": "system", "content": prompt},
        ]
        return llm.create_completion(
            response_model=SynthesizedResponse,
            messages=messages,
        )

    @staticmethod
    def convert_dataframe(
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
