# vector_db/api/models.py
from pydantic import BaseModel
from typing import List, Union

class SearchResult(BaseModel):
    vector_id: int
    distance: float
    metadata: Union[dict, str]

class SearchResults(BaseModel):
    results: List[SearchResult]

class SearchQuery(BaseModel):
    query_vector: List[float]
    k: int = 5