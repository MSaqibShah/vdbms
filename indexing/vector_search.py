import hnswlib
import numpy as np


class VectorSearch:
    def __init__(self, dimension, space='l2', ef_construction=200, M=16):
        """
        Initialize the HNSWlib-based Vector Database Management System.

        :param dimension: Dimensionality of vectors.
        :param space: Distance metric ('l2' for Euclidean, 'cosine' for Cosine similarity).
        :param ef_construction: Controls index construction accuracy.
        :param M: Maximum number of connections per element in the graph.
        """
        self.dimension = dimension
        self.space = space
        self.metadata_store = {}  # Dictionary for metadata management
        self.index = hnswlib.Index(space=space, dim=dimension)
        self.index.init_index(max_elements=10000,
                              ef_construction=ef_construction, M=M)
        self.index.set_ef(200)  # Controls search accuracy

    def add_vectors(self, vectors, metadata):
        """
        Add vectors to the index along with metadata.

        :param vectors: NumPy array of vectors to add (shape: [num_vectors, dimension]).
        :param metadata: List of metadata corresponding to each vector.
        """
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata length must match.")

        start_id = self.index.get_current_count()
        ids = np.arange(start_id, start_id + len(vectors))
        self.index.add_items(vectors, ids)

        # Add metadata
        for i, meta in zip(ids, metadata):
            self.metadata_store[i] = meta

    def search(self, query_vector, k=5):
        """
        Search for the top-k nearest neighbors of a query vector.

        :param query_vector: NumPy array of the query vector (shape: [dimension]).
        :param k: Number of nearest neighbors to return.
        :return: List of tuples (index, distance, metadata) for top-k neighbors.
        """
        labels, distances = self.index.knn_query(
            np.array(query_vector).reshape(1, -1), k=k)
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            if idx in self.metadata_store:
                results.append((idx, dist, self.metadata_store[idx]))
        return results

    def update_vector(self, vector_id, new_vector, new_metadata):
        """
        Update a vector and its metadata.

        :param vector_id: ID of the vector to update.
        :param new_vector: New vector data (shape: [dimension]).
        :param new_metadata: New metadata to associate with the vector.
        """
        self.delete_vector(vector_id)
        self.index.add_items(np.array([new_vector]).astype(
            'float32'), np.array([vector_id]))
        self.metadata_store[vector_id] = new_metadata

    def delete_vector(self, vector_id):
        """
        Delete a vector and its metadata from the index.

        :param vector_id: ID of the vector to delete.
        """
        if vector_id in self.metadata_store:
            del self.metadata_store[vector_id]
        else:
            raise ValueError(
                f"Vector ID {vector_id} not found in metadata store.")

        # Mark vector as deleted by recreating the index without it
        remaining_vectors = []
        remaining_metadata = []
        current_count = self.index.get_current_count()

        for idx in range(current_count):
            if idx != vector_id:
                remaining_vectors.append(self.index.get_items([idx])[0])
                remaining_metadata.append(self.metadata_store[idx])

        # Recreate the index
        self._recreate_index(
            np.array(remaining_vectors, dtype='float32'), remaining_metadata)

    def _recreate_index(self, vectors, metadata):
        """
        Recreate the HNSW index with new vectors and metadata.
        """
        max_elements = len(vectors) if len(vectors) > 0 else 10000
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.index.init_index(max_elements=max_elements,
                              ef_construction=200, M=16)
        self.index.set_ef(200)
        ids = np.arange(len(vectors))
        self.index.add_items(vectors, ids)

        # Rebuild metadata
        self.metadata_store = {i: metadata[i] for i in range(len(metadata))}

    def save_index(self, file_path):
        """
        Save the index and metadata to a file.

        :param file_path: Path to save the index file.
        """
        self.index.save_index(file_path)
        with open(file_path + ".meta", "w") as f:
            for vector_id, meta in self.metadata_store.items():
                f.write(f"{vector_id}\t{meta}\n")

    def load_index(self, file_path):
        """
        Load the index and metadata from a file.

        :param file_path: Path to load the index file.
        """
        # Create a new index object
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)

        # Load the index
        self.index.load_index(file_path)

        # Load metadata
        self.metadata_store.clear()
        try:
            with open(file_path + ".meta", "r") as f:
                for line in f:
                    vector_id, meta = line.strip().split("\t")
                    self.metadata_store[int(vector_id)] = meta
        except FileNotFoundError:
            print("Metadata file not found. Proceeding with empty metadata.")

    def get_vector_count(self):
        """
        Get the total number of vectors in the index.

        :return: Total vector count.
        """
        return self.index.get_current_count()


# Example Usage
if __name__ == "__main__":
    # Initialize VDBMS
    dbms = VectorSearch(dimension=128)

    # Add vectors and metadata
    vectors = np.random.random((10, 128)).astype('float32')
    metadata = [f"Vector {i}" for i in range(10)]
    dbms.add_vectors(vectors, metadata)

    # Search for nearest neighbors
    query = np.random.random(128).astype('float32')
    results = dbms.search(query, k=5)
    print("Search Results:", results)

    # Update a vector
    dbms.update_vector(0, np.random.random(
        128).astype('float32'), "Updated Vector 0")

    # Delete a vector
    dbms.delete_vector(1)

    # Save and load index
    dbms.save_index("hnsw_index")
    dbms.load_index("hnsw_index")