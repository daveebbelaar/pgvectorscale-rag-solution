# Building a High-Performance RAG Solution with Pgvectorscale and Python

This tutorial will guide you through setting up and using `pgvectorscale` with Docker and Python, leveraging OpenAI's powerful `text-embedding-3-small` model for embeddings. You'll learn to build a cutting-edge RAG (Retrieval-Augmented Generation) solution, combining advanced retrieval techniques (including hybrid search) with intelligent answer generation based on the retrieved context. Perfect for AI engineers looking to enhance their projects with state-of-the-art vector search and generation capabilities with the power of PostgreSQL.

## YouTube Tutorial
You can watch the full tutorial here on [YouTube](https://youtu.be/hAdEuDBN57g).

## Pgvectorscale Documentation

For more information about using PostgreSQL as a vector database in AI applications with Timescale, check out these resources:

- [GitHub Repository: pgvectorscale](https://github.com/timescale/pgvectorscale)
- [Blog Post: PostgreSQL and Pgvector: Now Faster Than Pinecone, 75% Cheaper, and 100% Open Source](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/)
- [Blog Post: RAG Is More Than Just Vector Search](https://www.timescale.com/blog/rag-is-more-than-just-vector-search/)
- [Blog Post: A Python Library for Using PostgreSQL as a Vector Database in AI Applications](https://www.timescale.com/blog/a-python-library-for-using-postgresql-as-a-vector-database-in-ai-applications/)

## Why PostgreSQL?

Using PostgreSQL with pgvectorscale as your vector database offers several key advantages over dedicated vector databases:

- PostgreSQL is a robust, open-source database with a rich ecosystem of tools, drivers, and connectors. This ensures transparency, community support, and continuous improvements.

- By using PostgreSQL, you can manage both your relational and vector data within a single database. This reduces operational complexity, as there's no need to maintain and synchronize multiple databases.

- Pgvectorscale enhances pgvector with faster search capabilities, higher recall, and efficient time-based filtering. It leverages advanced indexing techniques, such as the DiskANN-inspired index, to significantly speed up Approximate Nearest Neighbor (ANN) searches.

Pgvectorscale Vector builds on top of [pgvector](https://github.com/pgvector/pgvector), offering improved performance and additional features, making PostgreSQL a powerful and versatile choice for AI applications.

## Prerequisites

- Docker
- Python 3.7+
- OpenAI API key
- PostgreSQL GUI client
- Cohere API key (optional, for reranking)

## Steps

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client (I use TablePlus)
3. Create a Python script to insert document chunks as vectors using OpenAI embeddings
4. Create a Python function to perform similarity search

## Detailed Instructions

### 1. Set up Docker environment

Create a `docker-compose.yml` file with the following content:

```yaml
services:
  timescaledb:
    image: timescale/timescaledb-ha:pg16
    container_name: timescaledb
    environment:
      - POSTGRES_DB=postgres
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  timescaledb_data:
```

Run the Docker container:

```bash
docker compose up -d
```

### 2. Connect to the database using a PostgreSQL GUI client

- Open client
- Create a new connection with the following details:
  - Host: localhost
  - Port: 5432
  - User: postgres
  - Password: password
  - Database: postgres

### 3. Create a Python script to insert document chunks as vectors

See `insert_vectors.py` for the implementation. This script uses OpenAI's `text-embedding-3-small` model to generate embeddings.

### 4. Create a Python function to perform similarity search

See `similarity_search.py` for the implementation. This script also uses OpenAI's `text-embedding-3-small` model for query embedding.

## Usage

1. Create a copy of `example.env` and rename it to `.env`
2. Open `.env` and fill in your OpenAI API key. Leave the database settings as is
3. Run the Docker container
4. Install the required Python packages using `pip install -r requirements.txt`
5. Execute `insert_vectors.py` to populate the database
6. Play with `similarity_search.py` to perform similarity searches

## Using ANN search indexes to speed up queries

Timescale Vector offers indexing options to accelerate similarity queries, particularly beneficial for large vector datasets (10k+ vectors):

1. Supported indexes:
   - timescale_vector_index (default): A DiskANN-inspired graph index
   - pgvector's HNSW: Hierarchical Navigable Small World graph index
   - pgvector's IVFFLAT: Inverted file index

2. The DiskANN-inspired index is Timescale's latest offering, providing improved performance. Refer to the [Timescale Vector explainer blog](https://www.timescale.com/blog/pgvector-is-now-as-fast-as-pinecone-at-75-less-cost/) for detailed information and benchmarks.

For optimal query performance, creating an index on the embedding column is recommended, especially for large vector datasets.

## Cosine Similarity in Vector Search

### What is Cosine Similarity?

Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space. It's a measure of orientation rather than magnitude.

- Range: -1 to 1 (for normalized vectors, which is typical in text embeddings)
- 1: Vectors point in the same direction (most similar)
- 0: Vectors are orthogonal (unrelated)
- -1: Vectors point in opposite directions (most dissimilar)

### Cosine Distance

In pgvector, the `<=>` operator computes cosine distance, which is 1 - cosine similarity.

- Range: 0 to 2
- 0: Identical vectors (most similar)
- 1: Orthogonal vectors
- 2: Opposite vectors (most dissimilar)

### Interpreting Results

When you get results from similarity_search:

- Lower distance values indicate higher similarity.
- A distance of 0 would mean exact match (rarely happens with embeddings).
- Distances closer to 0 indicate high similarity.
- Distances around 1 suggest little to no similarity.
- Distances approaching 2 indicate opposite meanings (rare in practice).

## Keyword Search

Keyword search in PostgreSQL leverages full-text search capabilities to find relevant documents based on textual queries. Our implementation uses the `to_tsvector` and `websearch_to_tsquery` functions for efficient text searching.

### How it works

The `keyword_search` method in `vector_store.py` performs the following steps:

1. Converts the contents of each document into a tsvector (text search vector) using `to_tsvector('english', contents)`.
2. Transforms the user's query into a tsquery (text search query) using `websearch_to_tsquery('english', %s)`.
3. Matches the query against the document vectors using the `@@` operator.
4. Ranks the results using `ts_rank_cd` for relevance scoring.

Here's a breakdown of the SQL query used:

```sql
SELECT id, contents, ts_rank_cd(to_tsvector('english', contents), query) as rank
FROM {self.vector_settings.table_name}, websearch_to_tsquery('english', %s) query
WHERE to_tsvector('english', contents) @@ query
ORDER BY rank DESC
LIMIT %s
```

- `to_tsvector('english', contents)`: Converts the document content into a searchable vector of lexemes.
- `websearch_to_tsquery('english', %s)`: Parses the user's query into a tsquery, supporting web search syntax.
- `@@`: The match operator, returns true if the tsvector matches the tsquery.
- `ts_rank_cd`: Calculates the relevance score based on the frequency and proximity of matching terms.

### Advantages of Keyword Search

Keyword search is particularly useful for finding exact matches and specific terms that semantic models might miss. It excels at:

1. Locating precise phrases or technical terms.
2. Finding rare words or unique identifiers.
3. Matching acronyms or abbreviations.

While semantic search can understand context and meaning, keyword search ensures that specific, important terms are not overlooked, making it a valuable complement to semantic search in a hybrid approach.

### GIN Index

The GIN (Generalized Inverted Index) index is implemented in the VectorStore class to increase the performance of keyword searches within PostgreSQL. By creating an inverted index on the text content of documents, it enables rapid full-text search operations, allowing for efficient retrieval of relevant documents even in large datasets.

### What About BM25 Ranking?

While PostgreSQL's built-in ranking functions are powerful, they don't directly implement the BM25 (Best Matching 25) algorithm, which is considered state-of-the-art for many information retrieval tasks. BM25 takes into account term frequency, inverse document frequency, and document length normalization.

Although PostgreSQL doesn't natively support BM25, it can be approximated using custom functions or extensions. For example, the `pg_search` extension can be used to implement BM25-like functionality. Alternatively, you can create a custom ranking function that mimics BM25 behavior using PostgreSQL's plpgsql language. In Anthropic's recent [guide](https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb), they implement BM25 with Elasticsearch in parallel to a vector database.

If exact BM25 ranking is crucial for your application, you might consider using specialized search engines like Elasticsearch or implementing a custom solution on top of PostgreSQL with Paradedb's [pg_search](https://github.com/paradedb/paradedb).

## Hybrid Search

Hybrid search combines the strengths of both keyword-based and semantic (vector) search to provide more comprehensive and accurate results. The `hybrid_search` method in `vector_store.py` implements this approach.

### How it works

1. Perform keyword search using `keyword_search` method.
2. Perform semantic search using `semantic_search` method.
3. Combine the results from both searches.
4. Remove duplicates, prioritizing the first occurrence (which maintains the original order and search type).
5. Optionally rerank the combined results using Cohere's reranking model.

This approach allows us to capture both lexical matches (from keyword search) and semantic similarities (from vector search), providing a more robust search experience.

## Reranking

Reranking improves search relevance by reordering the result set from a retriever using a different model. It computes a relevance score between the query and each data object, sorting them from most to least relevant. This two-stage process ensures efficiency by retrieving relevant objects before reranking them. In our implementation, we use Cohere's reranking model to achieve this. This is an important step when you combine semantic search results with keyword search results.

### Cohere's Reranking Implementation

Cohere's reranking model is a state-of-the-art system designed to reorder a list of documents based on their relevance to a given query. Here's how we use it in our `_rerank_results` method:

1. We send the original query and the combined results from keyword and semantic search to Cohere's rerank API.
2. The API returns a reordered list of documents along with relevance scores.
3. We create a new DataFrame with the reranked results, including the original search type (keyword or semantic) and the new relevance scores.
4. The results are sorted by the relevance score in descending order.

This process allows us to leverage Cohere's advanced language understanding capabilities to further refine our search results, potentially surfacing the most relevant documents that might have been ranked lower in the initial search.

By combining keyword search, semantic search, and reranking, we create a powerful and flexible search system that can handle a wide range of queries and document types effectively.
