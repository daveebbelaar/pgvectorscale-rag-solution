import logging
import time
from typing import Any, List, Optional

import pandas as pd
from database.db_connection import get_db_connection
from database.vector_store import VectorStore
from psycopg.sql import SQL, Identifier
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    key: str = Field(..., description="The key within the metadata JSON to filter on")
    value: Any = Field(..., description="The value to filter by in the metadata")


class ColumnFilter(BaseModel):
    column: str = Field(..., description="The database column to filter on")
    value: Any = Field(..., description="The value to filter by for the column")


class VectorRetriever:
    """A class for retrieving similar vectors from the database."""

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the VectorRetriever.

        Args:
            vector_store: An instance of VectorStore for embedding generation.
        """
        self.vector_store = vector_store

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
            query: The search query.
            table_name: The name of the table to search in.
            metadata_filter: Optional filter for metadata.
            column_filter: Optional filter for a specific column.
            k: The number of results to return. Defaults to 3.

        Returns:
            A DataFrame containing the search results.
        """
        query_embedding = self.vector_store.get_embedding(query)
        start_time = time.time()

        sql_query, query_params = self._build_search_query(
            query_embedding, table_name, metadata_filter, column_filter, k
        )

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query, query_params)
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        logging.info(
            f"Similarity search on table '{table_name}' completed in {elapsed_time:.3f} seconds"
        )

        # Get column names dynamically
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {table_name} LIMIT 0")
                column_names = [
                    desc[0] for desc in cur.description if desc[0] != "embedding"
                ]

        return pd.DataFrame(results, columns=column_names + ["cosine_distance"])

    def _build_search_query(
        self,
        query_embedding: List[float],
        table_name: str,
        metadata_filter: Optional[MetadataFilter],
        column_filter: Optional[ColumnFilter],
        k: int,
    ) -> tuple[SQL, dict]:
        """
        Build the SQL query for similarity search.

        Args:
            query_embedding: The embedding of the search query.
            table_name: The name of the table to search in.
            metadata_filter: Optional filter for metadata.
            column_filter: Optional filter for a specific column.
            k: The number of results to return.

        Returns:
            A tuple containing the SQL query and its parameters.
        """
        # Get column names dynamically
        with get_db_connection() as conn:
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
