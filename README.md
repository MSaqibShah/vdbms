# Hybrid Vector Database Management System

## Overview
This project introduces a **Hybrid Vector Database Management System (VDBMS)** that bridges the gap between traditional relational databases and high-performance vector search. The system enables SQL-like querying of high-dimensional data, making vector search accessible to non-expert users.

## Features
- **SQL-like Query Interface**: Translate human-readable queries into vector operations.
- **Advanced Indexing**: Uses Hierarchical Navigable Small World (HNSW) indexing for efficient similarity searches.
- **Multi-Table Support**: Each dataset has its own dedicated HNSW index for optimized retrieval.
- **Hybrid Storage**: Supports relational metadata storage (SQLite/PostgreSQL) and vector storage (HNSWlib/FAISS).
- **Distance Metrics**: Supports Euclidean (L2) and Cosine similarity metrics.
- **Scalable & Performant**: Designed for AI-driven applications like semantic search and recommender systems.

## System Architecture
The system consists of three core layers:
1. **Query Processing Layer**: Parses SQL-like queries and translates them into vector operations.
2. **Indexing Layer**: Organizes and manages vector data using HNSW indexing.
3. **Storage Layer**: Manages vector storage (.hnsw files) and metadata storage (SQLite/PostgreSQL).

## Technologies Used
- **Programming Language**: Python
- **Database Management**: PostgreSQL + pgvector, SQLite
- **Vector Indexing**: HNSWlib, FAISS
- **API Framework**: FastAPI
- **Storage**: Local disk, Redis, or Cloud (S3)
- **Containerization**: Docker

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/hybrid-vector-db.git
   cd hybrid-vector-db
   ```
2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the API**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   **To test on terminal**
   ```bash
   python test_vector_db.py 
   ```
