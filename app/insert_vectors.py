import pandas as pd
from database.vector_store import VectorStore

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")

# Insert data into the database
vector_store = VectorStore()
vector_store.insert_data(df=df, table_name="embeddings")
