import psycopg
from config.settings import get_settings


def get_db_connection() -> psycopg.Connection:
    """
    Create and return a database connection.

    Returns:
        psycopg.Connection: A psycopg database connection.
    """
    settings = get_settings()
    return psycopg.connect(**settings.database.model_dump())
