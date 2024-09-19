import pandas as pd
from services.vector_service import VectorService

# Read the CSV file
df = pd.read_csv("../data/faq_data.csv", sep=";")

vector_service = VectorService()
vector_service.insert_data(df, "faq_embedding")
