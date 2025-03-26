import hnswlib
import sqlite3
import numpy as np
import os
import pickle
import hashlib
import time
import json  # For schema persistence
import shutil  # For deleting table directories from disk

class VectorDatabase:
    def __init__(self, dimension, store_path="./store"):
        """
        Initialize the vector database with HNSWlib and SQLite.

        :param dimension: Dimensionality of vectors.
        :param store_path: Path to the folder storing index and metadata.
        """
        self.dimension = dimension
        self.store_path = store_path
        self.db_path = os.path.join(store_path, "metadata.db")
        self.vector_index_path = os.path.join(store_path, "vectors.hnsw")
        self.binary_metadata_path = os.path.join(store_path, "metadata")

        # Initialize HNSWlib
        self.index = hnswlib.Index(space='l2', dim=dimension)
        self.index.init_index(max_elements=10000, ef_construction=200, M=16)
        self.index.set_ef(50)

        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
        if not os.path.exists(self.binary_metadata_path):
            os.makedirs(self.binary_metadata_path)

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
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata length must match.")
        for vector, metadata in zip(vectors, metadata_list):
            vector_id = hash(pickle.dumps(vector)) & 0x7FFFFFFF
            self.index.add_items(np.array([vector]), np.array([vector_id]))
            metadata_path = self._save_metadata(vector_id, metadata)
            self._insert_metadata_record(vector_id, metadata_path)

    def _save_metadata(self, vector_id, metadata):
        metadata_path = os.path.join(self.binary_metadata_path, f"metadata_{vector_id}.bin")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        return metadata_path

    def _insert_metadata_record(self, vector_id, metadata_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO metadata (vector_id, metadata_path) VALUES (?, ?)",
                       (vector_id, metadata_path))
        conn.commit()
        conn.close()

    def search(self, query_vector, k=5):
        labels, distances = self.index.knn_query(np.array(query_vector).reshape(1, -1), k=k)
        results = []
        for vector_id, distance in zip(labels[0], distances[0]):
            metadata = self._load_metadata(vector_id)
            results.append((vector_id, distance, metadata))
        return results

    def _load_metadata(self, vector_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT metadata_path FROM metadata WHERE vector_id={vector_id};")
        result = cursor.fetchone()
        conn.close()
        if not result:
            raise ValueError(f"Metadata for vector ID {vector_id} not found.")
        metadata_path = result[0]
        if not os.path.exists(metadata_path):
            raise ValueError(f"Metadata file for vector ID {vector_id} is missing.")
        with open(metadata_path, "rb") as f:
            return pickle.load(f)

    def update_metadata(self, vector_id, new_metadata):
        """
        Update the metadata file for a given vector.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_path FROM metadata WHERE vector_id=?", (vector_id,))
        result = cursor.fetchone()
        conn.close()
        if not result:
            raise ValueError(f"Metadata for vector ID {vector_id} not found.")
        metadata_path = result[0]
        with open(metadata_path, "wb") as f:
            pickle.dump(new_metadata, f)

    def save_index(self):
        self.index.save_index(self.vector_index_path)

    def load_index(self):
        self.index.load_index(self.vector_index_path)

    def get_vector_count(self):
        return self.index.get_current_count()

    def list_all(self):
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

    def delete_vector(self, vector_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT metadata_path FROM metadata WHERE vector_id=?", (vector_id,))
        result = cursor.fetchone()
        if result:
            metadata_path = result[0]
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            cursor.execute("DELETE FROM metadata WHERE vector_id=?", (vector_id,))
            conn.commit()
        conn.close()
        try:
            self.index.mark_deleted(vector_id)
        except Exception:
            pass

class Table:
    def __init__(self, name, schema_info, store_path):
        """
        :param name: Table name (e.g., 'users')
        :param schema_info: A dictionary with keys "columns" (list of [col_name, col_type])
                            and "dimension" (int or None).
        :param store_path: Directory where this table's data is stored.
        """
        self.name = name
        if isinstance(schema_info, list):
            schema_info = {"columns": schema_info, "dimension": None}
        self.schema = schema_info.get("columns", [])
        self.dimension = schema_info.get("dimension")
        self.table_id = self._generate_table_id()
        self.store_path = store_path
        if self.dimension is not None:
            self.vector_db = VectorDatabase(dimension=self.dimension, store_path=self.store_path)
            if os.path.exists(self.vector_db.vector_index_path):
                self.vector_db.load_index()
        else:
            self.vector_db = None

    def _generate_table_id(self):
        text_to_hash = f"{self.name}-{time.time()}"
        return hashlib.sha256(text_to_hash.encode()).hexdigest()

    def insert_vectors(self, vectors, metadata_list):
        if self.vector_db is None:
            self.dimension = vectors.shape[1]
            self.vector_db = VectorDatabase(dimension=self.dimension, store_path=self.store_path)
            # Update the persisted schema with the new dimension.
            schema_file = os.path.join(self.store_path, "schema.json")
            with open(schema_file, "r") as f:
                schema_info = json.load(f)
            if isinstance(schema_info, list):
                schema_info = {"columns": schema_info, "dimension": None}
            schema_info["dimension"] = self.dimension
            with open(schema_file, "w") as f:
                json.dump(schema_info, f)
        self.vector_db.add_vectors(vectors, metadata_list)

    def search_vectors(self, query_vector, k=5):
        if self.vector_db is None:
            return []
        return self.vector_db.search(query_vector, k=k)

    def get_vector_count(self):
        if self.vector_db is None:
            return 0
        return self.vector_db.get_vector_count()

    def list_all(self):
        if self.vector_db is None:
            return []
        return self.vector_db.list_all()

    def list_all_sorted_by(self, col, asc=True):
        all_records = self.list_all()
        def sort_key(rec):
            md = rec[1]
            return md.get(col, None)
        all_records.sort(key=sort_key, reverse=(not asc))
        return all_records

    def delete_records(self, column, value):
        if self.vector_db is None:
            return 0
        all_records = self.vector_db.list_all()
        delete_count = 0
        for vector_id, metadata in all_records:
            if str(metadata.get(column)) == str(value):
                self.vector_db.delete_vector(vector_id)
                delete_count += 1
        return delete_count

    def update_records(self, condition_column, condition_value, updates):
        """
        Update all records where metadata[condition_column] equals condition_value.
        :param condition_column: The column to check.
        :param condition_value: The value to match.
        :param updates: A dictionary of {column: new_value} to update.
        :return: Number of records updated.
        """
        if self.vector_db is None:
            return 0
        all_records = self.vector_db.list_all()  # returns list of (vector_id, metadata)
        update_count = 0
        for vector_id, metadata in all_records:
            if str(metadata.get(condition_column)) == str(condition_value):
                metadata.update(updates)
                # Re-write the metadata file
                file_path = os.path.join(self.vector_db.binary_metadata_path, f"metadata_{vector_id}.bin")
                with open(file_path, "wb") as f:
                    pickle.dump(metadata, f)
                update_count += 1
        return update_count

class TablesManager:
    def __init__(self, global_store_path="./store"):
        self.global_store_path = global_store_path
        self.tables = {}
        if not os.path.exists(self.global_store_path):
            os.makedirs(self.global_store_path)
        self._load_persisted_tables()

    def _load_persisted_tables(self):
        for item in os.listdir(self.global_store_path):
            table_dir = os.path.join(self.global_store_path, item)
            if os.path.isdir(table_dir):
                schema_file = os.path.join(table_dir, "schema.json")
                if os.path.exists(schema_file):
                    with open(schema_file, "r") as f:
                        schema_info = json.load(f)
                    if isinstance(schema_info, list):
                        schema_info = {"columns": schema_info, "dimension": None}
                    table_obj = Table(name=item, schema_info=schema_info, store_path=table_dir)
                    self.tables[item] = table_obj

    def create_table(self, name, schema):
        if name in self.tables:
            raise ValueError(f"Table '{name}' already exists!")
        table_path = os.path.join(self.global_store_path, name)
        os.makedirs(table_path, exist_ok=True)
        schema_info = {"columns": schema, "dimension": None}
        schema_file = os.path.join(table_path, "schema.json")
        with open(schema_file, "w") as f:
            json.dump(schema_info, f)
        table_obj = Table(name=name, schema_info=schema_info, store_path=table_path)
        self.tables[name] = table_obj
        return table_obj

    def drop_table(self, name):
        if name not in self.tables:
            raise ValueError(f"Table '{name}' does not exist!")
        shutil.rmtree(self.tables[name].store_path, ignore_errors=True)
        del self.tables[name]

    def get_table(self, name):
        if name not in self.tables:
            raise ValueError(f"Table '{name}' does not exist!")
        return self.tables[name]

    def show_tables(self):
        results = []
        for tname, tobj in self.tables.items():
            count = tobj.get_vector_count()
            results.append((tname, tobj.table_id, count, tobj.schema))
        return results

if __name__ == "__main__":
    # Example usage if vs.py is run directly.
    manager = TablesManager("./multi_tables")
    schema = [("id", "INT"), ("name", "TEXT")]
    users_table = manager.create_table("users", schema)
    print("Created table 'users' with ID:", users_table.table_id)
    vectors = np.random.random((3, 3)).astype('float32')
    metadata_list = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"}
    ]
    users_table.insert_vectors(vectors, metadata_list)
    print(f"Inserted {len(vectors)} vectors into 'users' table.")
    query = np.random.random(3).astype('float32')
    results = users_table.search_vectors(query, k=2)
    print("Nearest neighbors in 'users':", results)
    all_info = manager.show_tables()
    print("Tables info:\n", all_info)
    all_records = users_table.list_all_sorted_by("id", asc=False)
    print("All user records sorted by id desc:\n", all_records)
