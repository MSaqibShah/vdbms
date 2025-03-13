# # vector_db/query_processor/execution.py
# from ..storage.vector_store import VectorStore
# from ..storage.metadata_store import MetadataStore
# import numpy as np

# class QueryExecutor:
#     def __init__(self):
#         self.vector_store = VectorStore()
#         self.vector_store.load_index()
#         self.metadata_store = MetadataStore()

#     def execute_query(self, parsed_query):
#         table_name = parsed_query["table_name"]
#         columns = parsed_query["columns"]
#         where_clause = parsed_query["where_clause"]

#         vector_results = []
#         metadata_results = []

#         if where_clause and "similarity" in where_clause.lower():
#             # Extract vector and threshold
#             # ... (Logic to extract from where_clause) ...
#             query_vector = np.array([0.1, 0.2, 0.3]) #place holder
#             threshold = 0.5 #place holder
#             labels, distances = self.vector_store.search(query_vector)

#             vector_results = [(label, distance) for label, distance in zip(labels, distances) if distance < threshold]

#         if where_clause and "similarity" not in where_clause.lower():
#             #metadata based filter.
#             metadata_results = self.metadata_store.get_all_metadata()
#             #apply filter based on where clause.

#         # Combine and filter results
#         final_results = []
#         #logic to combine vector and metadata results.

#         return final_results


# vector_db/query_processor/execution.py
from storage.vector_database import VectorDatabase
import config
import numpy as np

class QueryExecutor:
    def __init__(self):
        self.db = VectorDatabase(dimension=config.VECTOR_DIM, store_path=config.STORE_PATH)

    def execute_query(self, parsed_query):
        table_name = parsed_query["table_name"]
        columns = parsed_query["columns"]
        where_clause = parsed_query["where_clause"]

        if where_clause and "similarity" in where_clause.lower():
            # Extract vector and threshold
            # ... (Logic to extract from where_clause) ...
            query_vector = np.array([0.1, 0.2, 0.3]) #place holder
            threshold = 0.5 #place holder

            results = self.db.search(query_vector, k=10) #get all the results.

            vector_results = [(vector_id, distance, metadata) for vector_id, distance, metadata in results if distance < threshold]
            #apply filter based on threshold.
            return vector_results
        else:
            #metadata based filter.
            #logic to get all metadata from db.
            #logic to apply filter based on where clause.
            return [] #placeholder.