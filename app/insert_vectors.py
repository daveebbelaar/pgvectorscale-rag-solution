from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from datasets import load_dataset
from timescale_vector.client import uuid_from_time

# Initialize VectorStore
vec = VectorStore()

# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
subset = dataset["train"].shuffle(seed=42).select(range(1000))
df = pd.DataFrame(subset)


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    Args:
        row (pandas.Series): A row from the dataset containing an 'article' column.

    Returns:
        pandas.Series: A series containing the prepared record for insertion.

    Note:
        This function uses the current time for the UUID. To use a specific time,
        create a datetime object and use uuid_from_time(your_datetime).
    """
    content = row["article"]
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # DiskAnnIndex
vec.create_keyword_search_index()  # GIN Index
vec.upsert(records_df)
