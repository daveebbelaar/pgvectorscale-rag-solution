from services.vector_service import VectorService

vector_service = VectorService()
query = "What are your shipping options?"


resulst = vector_service.search(
    query,
    table_name="faq_embedding",
    k=3,
)
