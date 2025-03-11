import hnswlib
import sqlite3
import numpy as np
import os
import pickle


class VectorDatabase:
    def __init__(self, dimension, store_path="./store"):
        """
        Initialize the vector database with HNSWlib and SQLite.

        :param dimension: Dimensionality of vectors.
        :param db_path: Path to the SQLite database for metadata.
        :param vector_index_path: Path to save the HNSWlib vector index.
        """
        self.dimension = dimension
        self.store_path = store_path
        self.db_path = store_path + "/metadata.db"
        self.vector_index_path = store_path + "/vectors.hnsw"
        self.binary_metadata_path = store_path + "/metadata"

        # Initialize HNSWlib
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.index.set_ef(50)

        # check if the store path exists
        if not os.path.exists(self.store_path):
            try:
                os.makedirs(self.store_path)
            except OSError as e:
                print(e)
                raise
        if not os.path.exists(self.binary_metadata_path):
            try:
                os.makedirs(self.binary_metadata_path)
            except OSError as e:
                print(e)
                raise

        # Initialize SQLite
        self._init_metadata_db()

    def _init_metadata_db(self):
        """Initialize the SQLite database for metadata."""
        print(self.db_path)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                vector_id INTEGER PRIMARY KEY,
                metadata_path TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_vectors(self, vectors, metadata_list):
        """
        Add vectors and their associated metadata.

        :param vectors: NumPy array of vectors to add (shape: [num_vectors, dimension]).
        :param metadata_list: List of metadata corresponding to each vector.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata length must match.")

        for vector, metadata in zip(vectors, metadata_list):
            # Generate a unique hashed ID for the vector
            # Use vector content to generate a unique hash
            vector_id = hash(pickle.dumps(vector))
            vector_id = vector_id & 0x7FFFFFFF

            # Add the vector to HNSWlib
            self.index.add_items(np.array([vector]), np.array([vector_id]))

            # Save metadata
            metadata_path = self._save_metadata(vector_id, metadata)
            self._insert_metadata_record(vector_id, metadata_path)

    def _save_metadata(self, vector_id, metadata):
        """
        Save metadata to a binary file.

        :param vector_id: Hashed integer ID of the vector.
        :param metadata: Metadata object to store.
        :return: Path to the saved metadata file.
        """
        metadata_path = f"{self.binary_metadata_path}/metadata_{vector_id}.bin"
        try:
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            return metadata_path
        except Exception as e:
            print(e)
            raise

    def _insert_metadata_record(self, vector_id, metadata_path):
        """
        Insert metadata record into the SQLite database.

        :param vector_id: Hashed integer ID of the vector.
        :param metadata_path: Path to the binary metadata file.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO metadata (vector_id, metadata_path) VALUES (?, ?)",
            (vector_id, metadata_path)
        )
        conn.commit()
        conn.close()

    def search(self, query_vector, k=5):
        """
        Search for the top-k nearest neighbors of a query vector.

        :param query_vector: NumPy array of the query vector (shape: [dimension]).
        :param k: Number of nearest neighbors to return.
        :return: List of tuples (vector_id, distance, metadata) for the top-k neighbors.
        """
        labels, distances = self.index.knn_query(
            np.array(query_vector).reshape(1, -1), k=k)

        results = []

        for vector_id, distance in zip(labels[0], distances[0]):
            metadata = self._load_metadata(vector_id)
            results.append((vector_id, distance, metadata))
        return results

    def _load_metadata(self, vector_id):
        """
        Load metadata from its binary file using vector ID.

        :param vector_id: Hashed integer ID of the vector.
        :return: Metadata object.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT metadata_path FROM metadata WHERE vector_id={vector_id};")

        result = cursor.fetchone()
        conn.close()
        if not result:
            raise ValueError(f"Metadata for vector ID {vector_id} not found.")

        metadata_path = result[0]
        if not os.path.exists(metadata_path):
            raise ValueError(
                f"Metadata file for vector ID {vector_id} is missing.")

        with open(metadata_path, "rb") as f:
            return pickle.load(f)

    def save_index(self):
        """Save the HNSWlib vector index to a file."""
        self.index.save_index(self.vector_index_path)

    def load_index(self):
        """Load the HNSWlib vector index from a file."""
        self.index.load_index(self.vector_index_path)

    def get_vector_count(self):
        """
        Get the total number of vectors in the index.

        :return: Total vector count.
        """
        return self.index.get_current_count()


# Example Usage
if __name__ == "__main__":
    # Initialize database
    db = VectorDatabase(dimension=128)

    # Add vectors and metadata
    vectors = np.random.random((10, 128)).astype('float32')
    metadata_list = [{"name": f"Vector {i}", "info": f"Info {i}"}
                     for i in range(10)]
    db.add_vectors(vectors, metadata_list)

    # Search
    print("Searching for nearest neighbors...")
    query = np.random.random(128).astype('float32')
    print("Query Vector:", query)
    results = db.search(query, k=5)
    print("Search Results:", results)

    # Save and load index
    db.save_index()
    db.load_index()
