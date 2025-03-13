# vector_db/api/router.py
from fastapi import APIRouter, HTTPException
from .models import SearchQuery, SearchResults, SearchResult
from storage.vector_database import VectorDatabase
import config
import numpy as np

router = APIRouter()
db = VectorDatabase(dimension=config.VECTOR_DIM, store_path=config.STORE_PATH)

@router.post("/search", response_model=SearchResults)
def search(query: SearchQuery):
    try:
        results = db.search(np.array(query.query_vector), k=query.k)
        search_results = [SearchResult(vector_id=vector_id, distance=distance, metadata=metadata)
                          for vector_id, distance, metadata in results]
        return SearchResults(results=search_results)
    except Exception as e:
        print(f"Error in /search: {e}")
        raise HTTPException(status_code=500, detail=str(e))