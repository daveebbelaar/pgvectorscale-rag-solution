# Building Advanced Hybrid RAG Pipelines with PostgreSQL

In this guide, we'll build an advanced Retrieval-Augmented Generation (RAG) pipeline using PostgreSQL with **hybrid search** and **reranking** capabilities. We'll implement a powerful system that leverages both keyword-based and semantic vector search, enhanced with Cohere's reranking endpoint, to provide highly relevant and accurate results. By the end of this tutorial, you'll have a flexible and efficient search solution capable of handling complex queries across diverse document types, setting the foundation for more sophisticated AI applications.

## Prerequisites

This tutorial builds upon my [previous guide](https://github.com/daveebbelaar/pgvectorscale-rag-solution/tree/setup) on setting up Pgvectorscale to build a high-performance RAG system. Before diving into this, make sure you’ve completed that setup, as it lays the foundation for the steps we’ll take here. In this repository, each branch contains a different, but related tutorial, so feel free to explore them as you progress.

- Docker
- Python 3.7+
- OpenAI API key
- PostgreSQL GUI client
- Cohere API key (optional, for reranking)

## Steps

1. Set up Docker environment
2. Connect to the database using a PostgreSQL GUI client (I use TablePlus)
3. Create a new virtual Python environment and install the `requirements.txt`
4. Create a Python script to insert document as vectors using OpenAI embeddings
5. Create a Python function to perform similarity search

## Getting Started

1. Create a copy of `example.env` and rename it to `.env`
2. Open `.env` and fill in your API keys. Leave the database settings as is
3. Run the Docker container with `docker compose up -d`
4. Connect to the database using your favorite PostgreSQL GUI (see settings below).
5. Create a new Python virutal environemnt for this project
6. Install the required Python packages using `pip install -r requirements.txt`
7. Check `app/config/settings.py` so you understand how this app is set up
8. Execute `insert_vectors.py` to populate the database
9. Play with `search.py` to perform similarity searches

## Database settings

How to connect to the database using a PostgreSQL GUI client:

- Host: localhost
- Port: 5432
- User: postgres
- Password: password
- Database: postgres

## Keyword Search

Keyword search in PostgreSQL leverages full-text search capabilities to find relevant documents based on textual queries. Our implementation uses the `to_tsvector` and `websearch_to_tsquery` functions for efficient text searching. More info on this [here](https://www.postgresql.org/docs/current/textsearch.html).

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

```SQL
CREATE INDEX IF NOT EXISTS index_name
ON table_name USING gin(to_tsvector('english', contents));
```

### What About BM25 Ranking?

While PostgreSQL's built-in ranking functions are powerful, they don't directly implement the BM25 (Best Matching 25) algorithm, which is considered state-of-the-art for many information retrieval tasks. BM25 takes into account term frequency, inverse document frequency, and document length normalization (TF-IDF-DL).

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

Reranking improves search relevance by reordering the result set from a retriever using a different model. It computes a relevance score between the query and each data object, sorting them from most to least relevant. This two-stage process ensures efficiency by retrieving relevant objects before reranking them. In our implementation, we use [Cohere's reranking model](https://cohere.com/blog/rerank) to achieve this. This is an important step when you combine semantic search results with keyword search results.

### Cohere's Reranking Implementation

Cohere's reranking model is a system designed to reorder a list of documents based on their relevance to a given query. Here's how we use it in our `_rerank_results` method:

1. We send the original query and the combined results from keyword and semantic search to Cohere's rerank API.
2. The API returns a reordered list of documents along with relevance scores that are computed with a large languge model.
3. We create a new DataFrame with the reranked results, including the original search type (keyword or semantic) and the new relevance scores.
4. The results are sorted by the relevance score in descending order.

This process allows us to leverage Cohere's advanced language understanding capabilities to further refine our search results, potentially surfacing the most relevant documents that might have been ranked lower in the initial search.

By combining keyword search, semantic search, and reranking, we create a powerful and flexible search system that can handle a wide range of queries and document types effectively.

## Further Reading

- https://www.anthropic.com/news/contextual-retrieval
- https://www.timescale.com/blog/postgresql-hybrid-search-using-pgvector-and-cohere/
- https://jkatz05.com/post/postgres/hybrid-search-postgres-pgvector/
- https://www.timescale.com/blog/build-search-and-rag-systems-on-postgresql-using-cohere-and-pgai/