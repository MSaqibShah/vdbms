import hnswlib
import sqlite3
import numpy as np
import os
import pickle
import hashlib
import time

class VectorDatabase:
    def __init__(self, dimension, store_path="./store"):
        """
        Initialize the vector database with HNSWlib and SQLite.

        :param dimension: Dimensionality of vectors.
        :param store_path: Path to the folder storing index and metadata.
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

        # Ensure store path and metadata folder exist
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
            # Generate a unique hashed ID for the vector based on its content
            vector_id = hash(pickle.dumps(vector))
            vector_id = vector_id & 0x7FFFFFFF

            # Add the vector to HNSWlib
            self.index.add_items(np.array([vector]), np.array([vector_id]))

            # Save metadata and record it in SQLite
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
            raise ValueError(f"Metadata file for vector ID {vector_id} is missing.")

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

    def list_all(self):
        """
        List all stored vectors' metadata.
        
        :return: A list of tuples (vector_id, metadata).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT vector_id, metadata_path FROM metadata")
        rows = cursor.fetchall()
        conn.close()

        results = []
        for vector_id, metadata_path in rows:
            try:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
            except Exception as e:
                metadata = f"Error loading metadata: {e}"
            results.append((vector_id, metadata))
        return results

# -------------------------------------------------------
# Table and TablesManager
# -------------------------------------------------------

class Table:
    """
    A single table with typed schema: a list of (col_name, col_type).
    Each table gets its own VectorDatabase once dimension is known.
    """
    def __init__(self, name, schema, store_path):
        """
        :param name: Table name, e.g. 'users'
        :param schema: e.g. [("id","INT"),("name","TEXT"),("created_at","DATETIME")]
        :param store_path: folder where this table's data is stored
        """
        self.name = name
        # We'll store the typed schema as a list of (col_name, col_type).
        # E.g. [("id","INT"),("title","TEXT"),("ts","DATETIME")]
        self.schema = schema

        self.table_id = self._generate_table_id()
        self.store_path = store_path
        self.dimension = None      # we only know dimension once we see the first vector
        self.vector_db = None      # lazy-init VectorDatabase

    def _generate_table_id(self):
        """
        Create a unique hash for the table (name + timestamp).
        """
        text_to_hash = f"{self.name}-{time.time()}"
        return hashlib.sha256(text_to_hash.encode()).hexdigest()

    def insert_vectors(self, vectors, metadata_list):
        """
        Insert an array of vectors plus a list of metadata records.
        The 'dimension' is determined at first insert if self.vector_db is None.
        """
        if self.vector_db is None:
            self.dimension = vectors.shape[1]
            self.vector_db = VectorDatabase(dimension=self.dimension, store_path=self.store_path)
        self.vector_db.add_vectors(vectors, metadata_list)

    def search_vectors(self, query_vector, k=5):
        """
        Perform a nearest-neighbor search using this table's vector_db.
        """
        if self.vector_db is None:
            return []
        return self.vector_db.search(query_vector, k=k)

    def get_vector_count(self):
        """
        How many vectors are stored in this table?
        """
        if self.vector_db is None:
            return 0
        return self.vector_db.get_vector_count()

    def list_all(self):
        """
        Return all metadata records as (vector_id, metadata).
        No vector search, just a table-scan from SQLite.
        """
        if self.vector_db is None:
            return []
        return self.vector_db.list_all()

    def list_all_sorted_by(self, col, asc=True):
        """
        Return all records sorted by metadata[col].
        """
        all_records = self.list_all()  # [(vid, metadata), ...]
        # Sort by metadata[col], if present
        def sort_key(rec):
            md = rec[1]  # the metadata dict
            return md.get(col, None)

        all_records.sort(key=sort_key, reverse=(not asc))
        return all_records

class TablesManager:
    """
    Manages multiple Table objects. Each table is stored in a subdir of global_store_path.
    """
    def __init__(self, global_store_path="./store"):
        self.global_store_path = global_store_path
        self.tables = {}  # {table_name: Table obj}

    def create_table(self, name, schema):
        """
        Create a new table with typed schema:
          e.g. [("id","INT"),("text","TEXT"),("created_at","DATETIME")]
        """
        if name in self.tables:
            raise ValueError(f"Table '{name}' already exists!")

        table_path = os.path.join(self.global_store_path, name)
        os.makedirs(table_path, exist_ok=True)
        table_obj = Table(name=name, schema=schema, store_path=table_path)
        self.tables[name] = table_obj
        return table_obj

    def drop_table(self, name):
        """
        Remove a table from memory. Optionally remove from disk too.
        """
        if name not in self.tables:
            raise ValueError(f"Table '{name}' does not exist!")
        # You could also remove table's data from disk:
        # import shutil
        # shutil.rmtree(self.tables[name].store_path, ignore_errors=True)
        del self.tables[name]

    def get_table(self, name):
        """
        Return the Table object for 'name'
        """
        if name not in self.tables:
            raise ValueError(f"Table '{name}' does not exist!")
        return self.tables[name]

    def show_tables(self):
        """
        Return a list of (table_name, table_id, row_count, schema)
        so the user can see existing tables.
        """
        results = []
        for tname, tobj in self.tables.items():
            count = tobj.get_vector_count()
            results.append((tname, tobj.table_id, count, tobj.schema))
        return results

# -------------------------------------------------------
# Example usage (if you run vs.py directly)
# -------------------------------------------------------
if __name__ == "__main__":
    # 1) Create a manager
    manager = TablesManager("./multi_tables")

    # 2) Create a typed table
    # e.g. columns: id INT, name TEXT
    schema = [("id", "INT"), ("name", "TEXT")]
    users_table = manager.create_table("users", schema)
    print("Created table 'users' with ID:", users_table.table_id)

    # 3) Insert some random 3D vectors for demonstration
    vectors = np.random.random((3, 3)).astype('float32')
    metadata_list = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"}
    ]
    users_table.insert_vectors(vectors, metadata_list)
    print(f"Inserted {len(vectors)} vectors into 'users' table.")

    # 4) Searching
    query = np.random.random(3).astype('float32')
    results = users_table.search_vectors(query, k=2)
    print("Nearest neighbors in 'users':", results)

    # 5) Show tables
    all_info = manager.show_tables()
    print("Tables info:\n", all_info)

    # 6) Listing & Sorting
    # Suppose we want to see all user records sorted by 'id' descending
    all_records = users_table.list_all_sorted_by("id", asc=False)
    print("All user records sorted by id desc:\n", all_records)
