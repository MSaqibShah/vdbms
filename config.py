# vector_db/config.py
import os
from dotenv import load_dotenv

load_dotenv()

VECTOR_DIM = int(os.getenv("VECTOR_DIM", 128))
STORE_PATH = os.getenv("STORE_PATH", "./store") # Path to store index and metadata
HNSW_SPACE = os.getenv("HNSW_SPACE", "l2")
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", 200))
HNSW_M = int(os.getenv("HNSW_M", 16))