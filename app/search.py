from database.vector_store import VectorStore
from services.synthesizer import Synthesizer

# Initialize VectorStore
vec = VectorStore()

query = "Is there any news related to London?"

# --------------------------------------------------------------
# Semantic search
# --------------------------------------------------------------

semantic_results = vec.semantic_search(query=query, limit=5)

# --------------------------------------------------------------
# Simple keyword search
# --------------------------------------------------------------

keyword_results = vec.keyword_search(query=query, limit=5)


# --------------------------------------------------------------
# Hybrid search
# --------------------------------------------------------------

hybrid_results = vec.hybrid_search(query=query, keyword_k=10, semantic_k=10)


# --------------------------------------------------------------
# Reranking
# --------------------------------------------------------------

reranked_results = vec.hybrid_search(
    query=query, keyword_k=10, semantic_k=10, rerank=True, top_n=5
)

# --------------------------------------------------------------
# Synthesize
# --------------------------------------------------------------

response = Synthesizer.generate_response(question=query, context=reranked_results)
print(response.answer)
