import pandas as pd
from services.vector_service import VectorService

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")

# Insert data into the database
vector_service = VectorService()
vector_service.insert_data(df=df, table_name="embeddings")
