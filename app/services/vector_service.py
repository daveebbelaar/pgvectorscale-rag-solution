import logging
import time
from typing import Any, Optional

import pandas as pd
import psycopg
from config.settings import get_settings
from openai import OpenAI
from psycopg.sql import SQL, Identifier
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    key: str = Field(..., description="The key within the metadata JSON to filter on")
    value: Any = Field(..., description="The value to filter by in the metadata")


class ColumnFilter(BaseModel):
    column: str = Field(..., description="The database column to filter on")
    value: Any = Field(..., description="The value to filter by for the column")


class VectorService:
    """
    A service for managing vector operations and database interactions.

    This class provides methods for generating embeddings, performing similarity
    searches, and inserting data into a vector database.
    """

    def __init__(self):
        """Initialize the VectorService with settings and OpenAI client."""
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model

    def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for the given text.

        Args:
            text (str): The input text to generate an embedding for.

        Returns:
            list[float]: A list of floats representing the embedding.
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

    def _get_db_connection(self) -> psycopg.Connection:
        """
        Create and return a database connection.

        Returns:
            psycopg.Connection: A psycopg database connection.
        """
        return psycopg.connect(**self.settings.database.model_dump())

    def search(
        self,
        query: str,
        table_name: str = "embeddings",
        metadata_filter: Optional[MetadataFilter] = None,
        column_filter: Optional[ColumnFilter] = None,
        k: int = 3,
    ) -> pd.DataFrame:
        """
        Perform similarity search and return results as a pandas DataFrame.

        Args:
            query (str): The search query.
            table_name (str): The name of the table to search in.
            metadata_filter (Optional[MetadataFilter], optional): Optional filter for metadata.
            column_filter (Optional[ColumnFilter], optional): Optional filter for a specific column.
            k (int, optional): The number of results to return. Defaults to 3.

        Returns:
            pd.DataFrame: A DataFrame containing the search results.
        """
        query_embedding = self.get_embedding(query)
        start_time = time.time()

        sql_query, query_params = self._build_search_query(
            query_embedding, table_name, metadata_filter, column_filter, k
        )

        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, query_params)
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        logging.info(
            f"Similarity search on table '{table_name}' completed in {elapsed_time:.3f} seconds"
        )

        # Get column names dynamically
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
                column_names = [
                    desc[0] for desc in cur.description if desc[0] != "embedding"
                ]

        return pd.DataFrame(results, columns=column_names + ["cosine_distance"])

    def _build_search_query(
        self,
        query_embedding: list[float],
        table_name: str,
        metadata_filter: Optional[MetadataFilter],
        column_filter: Optional[ColumnFilter],
        k: int,
    ) -> tuple[SQL, dict]:
        """
        Build the SQL query for similarity search.

        Args:
            query_embedding (list[float]): The embedding of the search query.
            table_name (str): The name of the table to search in.
            metadata_filter (Optional[MetadataFilter]): Optional filter for metadata.
            column_filter (Optional[ColumnFilter]): Optional filter for a specific column.
            k (int): The number of results to return.

        Returns:
            tuple[SQL, dict]: A tuple containing the SQL query and its parameters.
        """
        # Get column names dynamically
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
                column_names = [
                    desc[0] for desc in cur.description if desc[0] != "embedding"
                ]

        columns_str = ", ".join(column_names)
        sql_query = SQL(f"""
            SELECT {columns_str}, embedding <=> %(query_embedding)s::vector AS distance
            FROM {{table_name}}
            {{where_clause}}
            ORDER BY embedding <=> %(query_embedding)s::vector
            LIMIT %(limit)s
        """)

        query_params = {"query_embedding": query_embedding, "limit": k}
        where_clauses = []

        if metadata_filter:
            where_clauses.append(
                SQL("{} = %(metadata_value)s").format(Identifier(metadata_filter.key))
            )
            query_params.update(
                {
                    "metadata_value": str(metadata_filter.value),
                }
            )

        if column_filter:
            where_clauses.append(
                SQL("{} = %(column_value)s").format(Identifier(column_filter.column))
            )
            query_params["column_value"] = column_filter.value

        where_clause = (
            SQL("WHERE ") + SQL(" AND ").join(where_clauses)
            if where_clauses
            else SQL("")
        )

        formatted_query = sql_query.format(
            table_name=Identifier(table_name), where_clause=where_clause
        )

        return formatted_query, query_params

    def insert_data(self, df: pd.DataFrame, table_name: str = "embeddings") -> None:
        """
        Insert new data into the database.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the data to insert.
            table_name (str): The name of the table to insert data into.

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

        with self._get_db_connection() as conn:
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
