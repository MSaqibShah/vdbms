import shutil
import os
import json
import hashlib
from vs import VectorIndex
from sentence_transformers import SentenceTransformer


class Table:
    def __init__(self, table_path, embedding_model=None, dimension=None):

        self.table_path = table_path
        self.schema = None
        self.embedding_column = None
        self.dimension = dimension
        self.vector_index = None
        self._load_schema()
        self._init_vector_index()
        self._set_embedder(embedding_model)

    def _set_embedder(self, embedder):
        """
        Set the embedding function to be used for text embedding.
        """
        if embedder is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        elif callable(embedder):
            self.embedding_model = embedder

    def _load_schema(self):
        """
        Load the schema and configuration of the table from disk.
        """
        schema_file = os.path.join(self.table_path, "schema.json")
        if not os.path.exists(schema_file):
            raise FileNotFoundError(
                f"Schema file not found in table path: {self.table_path}")

        with open(schema_file, "r") as f:
            data = json.load(f)
            self.schema = data["schema"]
            self.embedding_column = data["embedding_column"]
            self.dimension = data["dimension"]

    def _init_vector_index(self):
        """
        Load or initialize the VectorIndex.
        """
        self.vector_index = VectorIndex(
            dimension=self.dimension, store_path=self.table_path)

    def get_schema(self):
        """
        Return the table's schema.
        """
        return self.schema

    def get_embedding_column(self):
        return self.embedding_column

    def compute_hash_id(self, text):
        """
        Compute SHA-256 hash as a 32-bit integer ID.
        """
        return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) & 0x7FFFFFFF

    def get_vector_index(self):
        return self.vector_index

    # CRUD operations
    def insert(self, record: dict):
        """
        Insert a new record. Embeds the embedding_column text using the user-defined embedder.

        :param record: A dict with fields matching schema.
        :return: vector_id
        """
        if not hasattr(self, "embedding_model"):
            raise Exception("Embedding model not set.")

        cleaned_record = {}

        for col in self.schema:
            if col not in record:
                raise ValueError(f"Missing column '{col}' in record.")

            raw_value = record[col]
            dtype = self.schema[col].lower()

            if dtype == "text":
                cleaned_value = str(raw_value).strip()
            elif dtype == "number":
                if isinstance(raw_value, str):
                    raw_value = raw_value.strip()
                    # Try casting to number
                    if "." in raw_value:
                        cleaned_value = float(raw_value)
                    else:
                        cleaned_value = int(raw_value)
                else:
                    cleaned_value = float(raw_value) if isinstance(
                        raw_value, float) else int(raw_value)
            else:
                raise ValueError(
                    f"Unsupported data type for column '{col}': {dtype}")

            cleaned_record[col] = cleaned_value

        # Get and validate embedding column
        text = cleaned_record[self.embedding_column]
        if not isinstance(text, str):
            raise ValueError("Embedding column must be of type text.")

        vector = self._encode_text(text)
        vector_id = self.compute_hash_id(text)

        self.vector_index.add_vectors([vector], [cleaned_record])
        return vector_id

    def select(self, filter_by=None, sort_by=None, ascending=True, query_text=None, limit=5):
        """
        Select records with optional filtering, sorting, and semantic search.

        :param filter_by: list of (key, op_func, val) tuples for flexible filtering.
        :param sort_by: column to sort by.
        :param ascending: sort direction (default: True).
        :param query_text: if provided, perform semantic vector search.
        :param limit: max number of records to return after filtering/sorting.
        :return: list of (vector_id, metadata) or (vector_id, distance, metadata)
        """
        results = []

        if query_text:
            if not hasattr(self, "embedding_model"):
                raise Exception("Embedding model not set.")
            query_vector = self.embedding_model.encode(
                query_text).astype("float32")
            k = min(limit, self.vector_index.get_vector_count())
            search_results = self.vector_index.search(
                query_vector, k=k)  # (id, distance, metadata)

            if filter_by:
                search_results = [
                    (vid, dist, meta)
                    for vid, dist, meta in search_results
                    if self._match_filter(meta, filter_by)
                ]

            if sort_by:
                search_results.sort(key=lambda x: x[2].get(
                    sort_by), reverse=not ascending)

            results = search_results

        else:
            all_data = self.vector_index.list_all()  # (id, metadata)

            if filter_by:
                all_data = [
                    (vid, meta)
                    for vid, meta in all_data
                    if self._match_filter(meta, filter_by)
                ]

            if sort_by:
                all_data.sort(key=lambda x: x[1].get(
                    sort_by), reverse=not ascending)

            results = all_data

        if limit is not None:
            results = results[:limit]

        return results

    def _match_filter(self, metadata, filters):
        """
        Apply complex filters (list of (key, op_func, val)) to a metadata dict.

        :param metadata: dict of metadata fields
        :param filters: list of (key, op_func, val)
        :return: True if all conditions match
        """
        for key, op_func, expected_val in filters:
            meta_val = metadata.get(key)

            # Convert meta_val if it's a string but expected a number
            if isinstance(expected_val, (int, float)) and isinstance(meta_val, str):
                try:
                    meta_val = float(
                        meta_val) if '.' in meta_val else int(meta_val)
                except ValueError:
                    return False

            # Apply comparison
            if not op_func(meta_val, expected_val):
                return False

        return True

    def _encode_text(self, text: str):
        return self.embedding_model.encode(text).astype("float32")

    def update(self, updated_fields: dict, filter_by=None, sort_by=None, ascending=True, query_text=None, limit=None) -> int:
        """
        Update records matching filter/sort/search criteria.

        :param updated_fields: Dictionary of new field values.
        :param filter_by: dict for filtering.
        :param sort_by: column name for sorting.
        :param ascending: sort order (default: True).
        :param query_text: perform semantic search if provided.
        :param limit: max number of rows to update.
        :return: Number of records updated.
        """
        if self.embedding_column in updated_fields:
            raise ValueError(
                "Embedding column cannot be updated. Delete and reinsert instead.")

        matched_records = self.select(
            filter_by=filter_by,
            sort_by=sort_by,
            ascending=ascending,
            query_text=query_text,
            limit=limit
        )

        count = 0
        for record in matched_records:
            vector_id = record[0]
            if self.vector_index.update_metadata(vector_id, updated_fields):
                count += 1

        print(f"[Update] ✅ Updated {count} record(s)")
        return count

    def delete(self, filter_by=None, sort_by=None, ascending=True, query_text=None, limit=None) -> int:
        """
        Delete records matching filter/sort/search criteria using select().
        """
        matched_records = self.select(
            filter_by=filter_by,
            sort_by=sort_by,
            ascending=ascending,
            query_text=query_text,
            limit=limit
        )

        count = 0
        for record in matched_records:
            vector_id = record[0]
            self.vector_index.delete_vector(vector_id)
            count += 1

        print(f"[Delete] 🗑️ Deleted {count} record(s)")
        return count


# table_manager.py


class TableManager:
    def __init__(self, database_path="./store"):
        self.database_path = database_path
        os.makedirs(self.database_path, exist_ok=True)

    def _get_table_path(self, table_name):
        return os.path.join(self.database_path, table_name)

    def _get_database_path(self, db_name):
        return os.path.join(self.database_path, db_name)

    def create_table(self, table_name, dimension, schema, embedding_column):
        """
        Create a new table with the given schema and embedding column.
        """
        db_path = self.database_path
        table_path = self._get_table_path(table_name)

        db_name = os.path.basename(db_path)

        if os.path.exists(table_path):
            raise Exception(
                f"Table '{table_name}' already exists in database '{db_name}'.")

        # os.makedirs(db_path, exist_ok=True)
        os.makedirs(table_path)

        # Validate schema
        if embedding_column not in schema:
            raise Exception("Embedding column must be defined in the schema.")
        if schema[embedding_column].lower() != "text":
            raise Exception(
                f"Embedding column must be of type 'text', got '{schema[embedding_column]}'.")
        for col, dtype in schema.items():
            if dtype.lower() not in ["text", "number"]:
                raise Exception(
                    f"Invalid data type for column '{col}'. Only 'text' and 'number' allowed.")

        # Save schema to disk
        schema_data = {
            "dimension": dimension,
            "schema": schema,
            "embedding_column": embedding_column
        }

        with open(os.path.join(table_path, "schema.json"), "w") as f:
            json.dump(schema_data, f, indent=2)

        # Load via Table class (which initializes VectorIndex)
        Table(table_path)

        print(
            f"✅ Table '{table_name}' created in database '{db_name}' with schema: {schema}")

    def drop_table(self, table_name):
        """
        Delete a table directory and all its contents.
        """
        table_path = os.path.join(self.database_path, table_name)
        if not os.path.exists(table_path):
            raise Exception(f"Table '{table_name}' does not exist.")

        shutil.rmtree(table_path)
        print(f"🗑️ Dropped table '{table_name}'.")

    def list_tables(self):
        """
        List all tables in the specified database along with their schema and ID.
        """
        db_path = self.database_path
        db_name = os.path.basename(db_path)
        if not os.path.exists(db_path):
            raise Exception(f"Database '{db_name}' does not exist.")

        tables = []

        for table_name in os.listdir(db_path):
            table_path = os.path.join(db_path, table_name)
            if not os.path.isdir(table_path):
                continue

            schema_file = os.path.join(table_path, "schema.json")
            if not os.path.exists(schema_file):
                continue  # Skip corrupted/incomplete tables

            try:
                with open(schema_file, "r") as f:
                    schema_data = json.load(f)
            except Exception as e:
                schema_data = {"error": f"Could not load schema: {str(e)}"}

            tables.append({
                # or hash(table_name) if you want hashed IDs
                "table_id": hash(table_name),
                "table_name": table_name,
                "schema": schema_data.get("schema", {})
            })

        return tables

    def get_table(self, table_name):
        """
        Get a Table object for the specified table name.
        """
        table_path = os.path.join(self.database_path, table_name)
        if not os.path.exists(table_path):
            raise Exception(f"Table '{table_name}' does not exist.")
        return Table(table_path)

    def _stable_hash(self, name):
        """Generate a stable SHA-256 hash for any given name (used as table_id)."""
        return hashlib.sha256(name.encode("utf-8")).hexdigest()

    def list_tables(self):
        """
        List all tables in the specified database along with their schema and hashed ID.
        """
        if not os.path.exists(self.database_path):
            raise Exception(
                f"Database directory '{self.database_path}' does not exist.")

        tables = []

        for table_name in os.listdir(self.database_path):
            table_path = os.path.join(self.database_path, table_name)
            if not os.path.isdir(table_path):
                continue

            schema_file = os.path.join(table_path, "schema.json")
            if not os.path.exists(schema_file):
                continue  # Skip invalid tables

            try:
                with open(schema_file, "r") as f:
                    schema_data = json.load(f)
            except Exception as e:
                schema_data = {"error": f"Could not load schema: {str(e)}"}

            tables.append({
                "table_id": self._stable_hash(table_name),
                "table_name": table_name,
                "schema": schema_data.get("schema", {})
            })

        return tables


if __name__ == "__main__":

    schema = {
        "title": "text",
        "description": "text",
        "price": "number"
    }
    databse_name = "shopdb"

    table_name = "products"

    # Create a new table
    # table_manager = TableManager()
    # table_manager.create_table(
    #     databse_name, table_name, dimension=384, schema=schema, embedding_column="description")

    # Dummy records
    records = [
        {
            "title": "Wireless Mouse",
            "description": "Ergonomic mouse with USB receiver and long battery life",
            "price": 799
        },
        {
            "title": "Mechanical Keyboard",
            "description": "Tactile keys with RGB lighting for typing enthusiasts",
            "price": 2499
        },
        {
            "title": "USB-C Hub",
            "description": "Connect multiple devices with one USB-C port",
            "price": 1299
        },
        {
            "title": "Laptop Stand",
            "description": "Aluminium stand for laptops up to 17 inches",
            "price": 999
        },
        {
            "title": "Noise Cancelling Headphones",
            "description": "Block distractions and enjoy deep sound quality",
            "price": 3999
        }
    ]

    # Load existing table (already created with embedding_column="description")
    table = Table("./store/shopdb/products")

    # Insert records(only if you're not inserting duplicates)
    # print("\n=== Inserting Products ===")
    # for r in records:
    #     vid = table.insert(r)
    #     print(f"Inserted {r['title']} with ID: {vid}")

    # 1. Select all
    # print("\n=== All Products ===")
    for vid, meta in table.select():
        print(f"{vid} -> {meta}")

    # # 2. Filter by price
    # print("\n=== Products with price = 999 ===")
    # for vid, meta in table.select(filter_by={"price": 999}, limit=1):
    #     print(f"{vid} -> {meta}")

    # # 3. Sort by price (ascending)
    # print("\n=== Products sorted by price (ascending) ===")
    # for vid, meta in table.select(sort_by="price", ascending=True):
    #     print(f"{vid} -> ₹{meta['price']} - {meta['title']}")

    # # 4. Filter by title + sort by price (descending)
    # print("\n=== Filtered by title 'Laptop Stand', sorted by price (desc) ===")
    # for vid, meta in table.select(filter_by={"title": "Laptop Stand"}, sort_by="price", ascending=False):
    #     print(f"{vid} -> ₹{meta['price']} - {meta['title']}")

    # Semantic search for "wireless device", return top 5 matches
    # results = table.select(
    #     query_text="Laptop stand", limit=5)
    # print("\n=== Semantic Search Results for 'Laptop stand")
    # for vid, dist, meta in results:
    #     print(f"{vid} -> {meta['title']} (Distance: {dist})")

    # Semantic + filter + limit
    # results = table.select(query_text="usb RGB",
    #                        filter_by={"price": 1299}, limit=2)
    # print("\n=== Semantic Search + Filter Results ===")
    # for vid, dist, meta in results:
    #     print(f"{vid} -> {meta['title']} (Distance: {dist})")

    # table.update({"price": 1799}, filter_by={"title": "Laptop Stand"})
    # print("\n=== Updated Products ===")
    # for vid, meta in table.select(filter_by={"title": "Laptop Stand"}):
    #     print(f"{vid} -> ₹{meta['price']} - {meta['title']}")

    # table.delete(filter_by={"title": "Wireless Mouse"})
    print("\n=== After Delete ===")
    for vid, meta in table.select():
        print(f"{vid} -> {meta}")
