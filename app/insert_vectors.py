import uuid
from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")


# Prepare data for insertion
def prepare_record(row):
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid.uuid1()),
            "metadata": {
                "category": row["category"],
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
vec.upsert(records_df)
