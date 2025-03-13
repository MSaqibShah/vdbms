# vector_db/test_vector_db.py
import numpy as np
from storage.vector_database import VectorDatabase
import config

db = VectorDatabase(dimension=config.VECTOR_DIM, store_path=config.STORE_PATH)

vectors = np.random.random((10, config.VECTOR_DIM)).astype('float32')
metadata_list = [{"name": f"Vector {i}", "info": f"Info {i}"} for i in range(10)]

db.add_vectors(vectors, metadata_list)

query_vector = np.random.random(config.VECTOR_DIM).astype('float32')
results = db.search(query_vector, k=5)

print("Search Results:")
for vector_id, distance, metadata in results:
    print(f"Vector ID: {vector_id}, Distance: {distance}, Metadata: {metadata}")