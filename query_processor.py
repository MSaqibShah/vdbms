import os
import re
import numpy as np
from VectorSearch import VectorSearch
from tabulate import tabulate
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    def __init__(self, index_path="./store/vectors.hnsw"):
        self.index_path = index_path
        self.db = None
        self.dimension = None
        self.tables = {}

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._load()

    def _save(self):
        if self.db:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.db.save_index(self.index_path)
            with open(self.index_path + ".config", "w") as f:
                f.write(str(self.dimension))

    def _load(self):
        config_path = self.index_path + ".config"
        if not (os.path.exists(self.index_path) and os.path.exists(self.index_path + ".meta") and os.path.exists(config_path)):
            return

        try:
            with open(config_path, "r") as f:
                self.dimension = int(f.read().strip())

            print(f"ℹ️ Loading existing vector index with dimension {self.dimension}")
            self.db = VectorSearch(dimension=self.dimension)
            self.db.load_index(self.index_path)

        except Exception as e:
            print(f"⚠️ Failed to load index: {e}")

    def execute_sql(self, sql):
        sql_str = sql.strip()

        if sql_str.upper().startswith("CREATE TABLE"):
            return self._handle_create_table(sql_str)
        elif "INSERT INTO" in sql_str.upper() and "TEXT" in sql_str.upper():
            return self._handle_insert_text(sql_str)
        elif sql_str.upper().startswith("INSERT INTO"):
            return self._handle_insert(sql_str)
        elif "WHERE TEXT =" in sql_str.upper():
            return self._handle_semantic_select(sql_str)
        elif sql_str.strip().upper() == "SELECT * FROM VECTORS;":
            return self._handle_list_all(sql_str)
        elif sql_str.upper().startswith("SELECT"):
            return self._handle_select(sql_str)
        else:
            return "❌ Unsupported SQL statement."

    def _handle_create_table(self, sql):
        pattern = r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)"
        match = re.match(pattern, sql, re.IGNORECASE)

        if match:
            table_name = match.group(1)
            columns_str = match.group(2)
            columns = [col.strip() for col in columns_str.split(',') if col.strip()]
            self.tables[table_name.lower()] = columns
            return f"✅ Table '{table_name}' created with columns: {columns}"
        else:
            return "❌ Invalid CREATE TABLE statement."

    def _handle_insert(self, sql):
        pattern = r"INSERT\s+INTO\s+(\w+)\s*\(\s*vector\s*,\s*metadata\s*\)\s*VALUES\s*\(\s*\[(.*?)\]\s*,\s*'(.*?)'\s*\);?"
        match = re.match(pattern, sql, re.IGNORECASE)

        if not match:
            return "❌ Invalid INSERT INTO statement format."

        table_name = match.group(1)
        vector_str = match.group(2)
        metadata_str = match.group(3)

        try:
            vector_vals = [float(x.strip()) for x in vector_str.split(',')]
            vector_np = np.array([vector_vals], dtype='float32')

            if self.db is None:
                self.dimension = len(vector_vals)
                self.db = VectorSearch(dimension=self.dimension)
                print(f"ℹ️ Inferred vector dimension: {self.dimension}")

            elif len(vector_vals) != self.dimension:
                return f"❌ Dimension mismatch! Expected {self.dimension}, got {len(vector_vals)}."

            self.db.add_vectors(vector_np, [metadata_str])
            self._save()
            return f"✅ Inserted vector with metadata: '{metadata_str}'"

        except Exception as e:
            return f"⚠️ Error while parsing or inserting vector: {e}"

    def _handle_select(self, sql):
        pattern = r"""
            SELECT\s+\*\s+FROM\s+(\w+)\s+
            WHERE\s+VECTOR\s*=\s*\[(.*?)\]
            (?:\s+AND\s+METADATA\s*=\s*'(.*?)')?
            \s+LIMIT\s+(\d+);?
        """
        match = re.match(pattern, sql, re.IGNORECASE | re.VERBOSE)

        if not match:
            return "❌ Invalid SELECT statement format."

        table_name = match.group(1)
        vector_str = match.group(2)
        metadata_filter = match.group(3)
        limit_str = match.group(4)

        try:
            vector_vals = [float(x.strip()) for x in vector_str.split(',')]

            if self.db is None:
                return "⚠️ No vectors inserted yet. Cannot search."

            if len(vector_vals) != self.dimension:
                return f"❌ Dimension mismatch! Expected {self.dimension}, got {len(vector_vals)}."

            query_vector = np.array(vector_vals, dtype='float32')
            k = int(limit_str)

            vector_count = self.db.get_vector_count()
            if vector_count == 0:
                return "⚠️ Vector index is empty."

            if k > vector_count:
                print(f"ℹ️ Requested {k} neighbors, but only {vector_count} vectors exist. Reducing limit to {vector_count}.")
                k = vector_count

            results = self.db.search(query_vector, k=k)

            if metadata_filter:
                results = [r for r in results if r[2] == metadata_filter]
                if not results:
                    return f"⚠️ No results found matching metadata = '{metadata_filter}'."

            headers = ["Vector ID", "Distance", "Metadata"]
            table = [[i, round(d, 4), m] for i, d, m in results]
            return tabulate(table, headers=headers, tablefmt="pretty")

        except Exception as e:
            return f"⚠️ Error while searching: {e}"

    def _handle_semantic_select(self, sql):
        pattern = r"SELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\s+TEXT\s*=\s*'(.*?)'\s+LIMIT\s+(\d+);?"
        match = re.match(pattern, sql, re.IGNORECASE)

        if not match:
            return "❌ Invalid TEXT-based SELECT syntax."

        table_name = match.group(1)
        text_query = match.group(2)
        limit = int(match.group(3))

        if self.db is None:
            return "⚠️ No vectors in index yet."

        try:
            vector_query = self.embedding_model.encode(text_query).astype("float32")

            if self.dimension is None:
                self.dimension = len(vector_query)
                self.db = VectorSearch(dimension=self.dimension)
            elif len(vector_query) != self.dimension:
                return f"❌ Embedding dimension mismatch! Expected {self.dimension}, got {len(vector_query)}."

            total = self.db.get_vector_count()
            if total == 0:
                return "⚠️ Vector index is empty."

            if limit > total:
                print(f"ℹ️ Reducing LIMIT to {total} (only that many vectors exist).")
                limit = total

            results = self.db.search(vector_query, k=limit)

            headers = ["Vector ID", "Distance", "Metadata"]
            table = [[i, round(d, 4), m] for i, d, m in results]
            return tabulate(table, headers=headers, tablefmt="pretty")

        except Exception as e:
            return f"⚠️ Error during semantic search: {e}"

    def _handle_insert_text(self, sql):
        pattern = r"INSERT\s+INTO\s+(\w+)\s*\(\s*text\s*,\s*metadata\s*\)\s*VALUES\s*\(\s*'(.*?)'\s*,\s*'(.*?)'\s*\);?"
        match = re.match(pattern, sql, re.IGNORECASE)

        if not match:
            return "❌ Invalid INSERT TEXT syntax."

        table_name = match.group(1)
        text = match.group(2)
        metadata = match.group(3)

        try:
            vector = self.embedding_model.encode(text).astype("float32")
            vector_np = np.array([vector])

            if self.db is None:
                self.dimension = len(vector)
                self.db = VectorSearch(dimension=self.dimension)
                print(f"ℹ️ Inferred vector dimension from text: {self.dimension}")
            elif len(vector) != self.dimension:
                return f"❌ Embedding dimension mismatch! Expected {self.dimension}, got {len(vector)}."

            self.db.add_vectors(vector_np, [metadata])
            self._save()
            return f"✅ Inserted embedded vector from text: '{text}'"

        except Exception as e:
            return f"⚠️ Error embedding and inserting: {e}"

    def _handle_list_all(self, sql):
        pattern = r"SELECT\s+\*\s+FROM\s+(\w+);?"
        match = re.match(pattern, sql.strip(), re.IGNORECASE)

        if not match:
            return "❌ Invalid SELECT ALL statement."

        if self.db is None:
            return "⚠️ No vectors in memory."

        try:
            count = self.db.get_vector_count()
            if count == 0:
                return "⚠️ No vectors stored."

            rows = []
            for i in range(count):
                if i in self.db.metadata_store:
                    rows.append([i, self.db.metadata_store[i]])

            if not rows:
                return "⚠️ No metadata found for any vectors."

            return tabulate(rows, headers=["Vector ID", "Metadata"], tablefmt="pretty")

        except Exception as e:
            return f"⚠️ Error while listing all vectors: {e}"

if __name__ == "__main__":
    print("\U0001f9e0 Vector SQL Processor (Auto-infers dimensions on first insert)")
    print("Supported: CREATE TABLE, INSERT INTO, SELECT, SELECT ... WHERE TEXT = '<query>'")
    print("Type 'exit' to quit.\n")

    processor = QueryProcessor()

    while True:
        try:
            user_input = input("SQL> ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("\U0001f44b Exiting SQL processor.")
                break

            result = processor.execute_sql(user_input)
            print(result)
        except KeyboardInterrupt:
            print("\n\U0001f44b Exiting SQL processor.")
            break
        except Exception as e:
            print(f"⚠️ Runtime Error: {str(e)}")
