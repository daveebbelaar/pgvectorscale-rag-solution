from services.vector_service import VectorService
from services.synthesizer import Synthesizer

vector_service = VectorService()

# --------------------------------------------------------------
# Shipping question
# --------------------------------------------------------------

relevant_question = "What are your shipping options?"


results = vector_service.search(
    relevant_question,
    table_name="embeddings",
    k=3,
)


response = Synthesizer.generate_response(question=relevant_question, context=results)

print(f"\n{response.answer}")
print("Thought process:\n")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")

# --------------------------------------------------------------
# Irrelevant question
# --------------------------------------------------------------

irrelevant_question = "What is the weather in Tokyo?"

results = vector_service.search(
    irrelevant_question,
    table_name="embeddings",
    k=3,
)

response = Synthesizer.generate_response(question=irrelevant_question, context=results)
print(f"\n{response.answer}")
print("Thought process:\n")
for thought in response.thought_process:
    print(f"- {thought}")
print(f"\nContext: {response.enough_context}")
