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
