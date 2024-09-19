from database.vector_store import VectorStore
from database.vector_retriever import ColumnFilter, VectorRetriever
from services.synthesizer import Synthesizer

vector_store = VectorStore()
vector_retriever = VectorRetriever(vector_store)

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "What are your shipping options?"

results = vector_retriever.search(
    relevant_question,
    table_name="embeddings",
    k=3,
)

response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vector_retriever.search(
    irrelevant_question,
    table_name="embeddings",
    k=3,
)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)

print(f"\n{response.answer}")
print("\nThought process:")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")


# --------------------------------------------------------------
# Column filtering
# --------------------------------------------------------------

column_filter = ColumnFilter(column="category", value="Shipping")

results = vector_retriever.search(
    relevant_question,
    table_name="embeddings",
    column_filter=column_filter,
    k=3,
)
