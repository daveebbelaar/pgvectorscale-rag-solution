import logging
import time
from typing import List

import pandas as pd
from config.settings import get_settings
from database.db_connection import get_db_connection
from openai import OpenAI


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings and OpenAI client."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def insert_data(self, df: pd.DataFrame, table_name: str = "embeddings") -> None:
        """
        Insert new data into the database.

        Args:
            df: A pandas DataFrame containing the data to insert.
            table_name: The name of the table to insert data into.

        Raises:
            ValueError: If the input is not a DataFrame or missing required columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Get all columns except 'embedding'
        columns = [col for col in df.columns if col != "embedding"]

        # Generate embeddings for the first text column (assuming it's the most relevant)
        df["embedding"] = df[columns[0]].apply(self.get_embedding)

        # Prepare data for insertion
        tuple_data = [
            tuple(row[columns].tolist() + [row["embedding"]])
            for _, row in df.iterrows()
        ]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                placeholders = ", ".join(["%s"] * (len(columns) + 1))
                columns_str = ", ".join(columns + ["embedding"])
                insert_query = (
                    f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                )
                cur.executemany(insert_query, tuple_data)
            conn.commit()

        logging.info(
            f"Successfully inserted {len(tuple_data)} records into {table_name}"
        )
