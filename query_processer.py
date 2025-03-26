import re
import time
import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate

from vs import TablesManager

class QueryProcessor:
    """
    A SQL-like query processor that uses a multi-table vector system (vs.py).
    Ensures all metadata is stored as a dictionary, so `list_all_sorted_by(...)`
    won't fail with 'str' object has no attribute 'get'.
    """

    def __init__(self):
        """
        Initialize with a global TablesManager at ./store,
        plus a sentence_transformers embedding model for textual columns.
        """
        self.tables_manager = TablesManager(global_store_path="./store")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def execute_sql(self, sql):
        """
        Main dispatcher: parse the user's SQL and route to the appropriate handler.
        """
        sql_str = sql.strip()

        # 1) CREATE TABLE
        if sql_str.upper().startswith("CREATE TABLE"):
            return self._handle_create_table(sql_str)

        # 2) SHOW TABLES
        elif sql_str.upper() == "SHOW TABLES;":
            return self._handle_show_tables()

        # 3) SELECT ALL ... ORDER BY ...
        elif re.match(r"SELECT\s+ALL\s+FROM\s+\w+\s+ORDER\s+BY\s+\w+", sql_str, re.IGNORECASE):
            return self._handle_select_all_orderby(sql_str)

        # 4) SELECT (semantic search)
        elif re.search(r"WHERE\s+TEXT\s*=\s*'.*?'(\s+LIMIT\s+\d+)?", sql_str, re.IGNORECASE):
            return self._handle_semantic_select(sql_str)

        # 5) INSERT typed
        elif re.match(r"INSERT\s+INTO\s+\w+\s*\(.*?\)\s*VALUES\s*\(.*?\)", sql_str, re.IGNORECASE):
            # We'll check if it specifically uses (text, metadata), or typed columns
            # If it has "text, metadata" => _handle_insert_text
            # else => _handle_insert_typed
            if re.search(r"\(\s*text\s*,\s*metadata\s*\)", sql_str, re.IGNORECASE):
                return self._handle_insert_text(sql_str)
            else:
                return self._handle_insert_typed(sql_str)

        else:
            return "❌ Unsupported or invalid SQL statement."

    # ----------------------------------------------------------------------
    # 1) CREATE TABLE
    # ----------------------------------------------------------------------
    def _handle_create_table(self, sql):
        """
        CREATE TABLE <table_name> (col1 INT, col2 TEXT, col3 DATETIME, ...)
        We'll parse columns into [("col1","INT"),("col2","TEXT"), ...].
        """
        pattern = r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid CREATE TABLE statement."

        table_name = match.group(1)
        columns_str = match.group(2).strip()

        col_defs = [c.strip() for c in columns_str.split(',') if c.strip()]  # ["id INT","name TEXT","created_at DATETIME"]
        schema = []
        for col_def in col_defs:
            parts = col_def.split()
            if len(parts) != 2:
                return f"❌ Invalid column definition: '{col_def}'. Must be 'colName colType'."
            col_name, col_type = parts
            schema.append((col_name, col_type.upper()))

        try:
            self.tables_manager.create_table(table_name, schema)
            return f"✅ Table '{table_name}' created with columns: {schema}"
        except ValueError as ve:
            return str(ve)

    # ----------------------------------------------------------------------
    # 2) SHOW TABLES
    # ----------------------------------------------------------------------
    def _handle_show_tables(self):
        """
        SHOW TABLES;
        """
        info = self.tables_manager.show_tables()
        if not info:
            return "⚠️ No tables found."
        # info is [(table_name, table_id, row_count, [(col1,type1),...])]

        headers = ["Table Name", "Table ID (sha256)", "Row Count", "Schema"]
        return tabulate(info, headers=headers, tablefmt="pretty")

    # ----------------------------------------------------------------------
    # 3) SELECT ALL ... ORDER BY ...
    # ----------------------------------------------------------------------
    def _handle_select_all_orderby(self, sql):
        """
        SELECT ALL FROM <table> ORDER BY <col> [ASC|DESC];
        We'll do a table-scan, sorting by metadata[col].
        """
        pattern = r"SELECT\s+ALL\s+FROM\s+(\w+)\s+ORDER\s+BY\s+(\w+)\s*(ASC|DESC)?;?"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid SELECT ALL syntax. Try: SELECT ALL FROM table ORDER BY col ASC|DESC;"

        table_name = match.group(1)
        order_col = match.group(2)
        order_dir = match.group(3) if match.group(3) else "ASC"

        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"

        asc = (order_dir.upper() == "ASC")
        all_records = table_obj.list_all_sorted_by(order_col, asc=asc)
        if not all_records:
            return "⚠️ No records in this table."

        rows = []
        for (vid, meta) in all_records:
            # meta is guaranteed to be a dict thanks to our approach
            row_str = f"VID={vid}, {meta}"
            rows.append([row_str])

        return tabulate(rows, headers=["Record"], tablefmt="pretty")

    # ----------------------------------------------------------------------
    # 4) SELECT (semantic search)
    # ----------------------------------------------------------------------
    def _handle_semantic_select(self, sql):
        """
        SELECT * FROM <table> WHERE TEXT = 'search query' LIMIT n;
        We'll embed 'search query' => vector, then do ANN search.
        """
        pattern = r"SELECT\s+\*\s+FROM\s+(\w+)\s+WHERE\s+TEXT\s*=\s*'(.*?)'\s+LIMIT\s+(\d+);?"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid TEXT-based SELECT syntax."

        table_name = match.group(1)
        text_query = match.group(2)
        limit_str = match.group(3)

        try:
            limit = int(limit_str)
            query_vector = self.embedding_model.encode(text_query).astype("float32")

            table_obj = self.tables_manager.get_table(table_name)
            vector_count = table_obj.get_vector_count()
            if vector_count == 0:
                return f"⚠️ No vectors exist in the '{table_name}' table."

            if limit > vector_count:
                print(f"ℹ️ Reducing LIMIT from {limit} to {vector_count} (only {vector_count} vectors exist).")
                limit = vector_count

            results = table_obj.search_vectors(query_vector, k=limit)
            if not results:
                return "⚠️ No results found."

            headers = ["Vector ID", "Distance", "Metadata"]
            table = [[vid, round(dist, 4), meta] for (vid, dist, meta) in results]
            return tabulate(table, headers=headers, tablefmt="pretty")
        except ValueError as ve:
            return f"⚠️ {ve}"
        except Exception as e:
            return f"⚠️ Error during semantic search: {e}"

    # ----------------------------------------------------------------------
    # 5) INSERT TEXT (Legacy) => store as a dict, to avoid 'str' metadata
    # ----------------------------------------------------------------------
    def _handle_insert_text(self, sql):
        """
        INSERT INTO <table_name> (text, metadata) VALUES ('some text', 'some metadata');
        We'll embed the 'text' => vector, then store metadata as a dict: { "text_col": <string>, "user_metadata": <string> }.
        """
        pattern = r"INSERT\s+INTO\s+(\w+)\s*\(\s*text\s*,\s*metadata\s*\)\s*VALUES\s*\(\s*'(.*?)'\s*,\s*'(.*?)'\s*\);?"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid INSERT TEXT syntax."

        table_name = match.group(1)
        text_value = match.group(2)
        user_meta_str = match.group(3)

        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"

        # embed text => vector
        vector = self.embedding_model.encode(text_value).astype("float32")
        vectors = np.array([vector])

        # always store metadata as a dict
        metadata_dict = {
            "text": text_value,
            "user_metadata": user_meta_str
        }

        table_obj.insert_vectors(vectors, [metadata_dict])
        return f"✅ Inserted embedded vector from text '{text_value}' into table '{table_name}'"

    # ----------------------------------------------------------------------
    # 6) INSERT TYPED => store entire row_data as a dict
    # ----------------------------------------------------------------------
    def _handle_insert_typed(self, sql):
        """
        Example:
          INSERT INTO table (id, title, abstract) VALUES (101, 'My Paper', 'Text to embed' );
        We'll parse typed columns/values, build a dict for row_data, embed if there's a TEXT column, etc.
        """
        pattern = r"INSERT\s+INTO\s+(\w+)\s*\(\s*(.*?)\s*\)\s*VALUES\s*\(\s*(.*?)\s*\);?"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid INSERT statement format."

        table_name = match.group(1)
        col_str = match.group(2)
        val_str = match.group(3)

        columns = [c.strip() for c in col_str.split(',') if c.strip()]
        vals = self._parse_values(val_str)
        if len(columns) != len(vals):
            return f"❌ Column/Value count mismatch: {len(columns)} columns, {len(vals)} values."

        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"

        # Build typed row_data => a dictionary
        row_data = self._build_row_data(table_obj, columns, vals)
        self._auto_fill_missing_cols(table_obj, row_data)

        # Possibly embed if there's a text column
        # Option 2 approach: if no text => disallow
        try:
            vector = self._must_embed(table_obj, row_data)
        except ValueError as ve:
            return f"❌ Insert disallowed: {ve}"

        # Insert vector + row_data
        vectors = np.array([vector], dtype="float32")
        table_obj.insert_vectors(vectors, [row_data])

        return f"✅ Inserted typed record into table '{table_name}' with columns {columns}"

    # ----------------------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------------------
    def _parse_values(self, val_str):
        """
        Splits a comma-separated list of values, tries to interpret int/float, or remove quotes for strings.
        E.g. "123, 'Alice', 45.67, 'Hello world'"
        """
        parts = [p.strip() for p in val_str.split(',')]
        cleaned = []
        for p in parts:
            if p.startswith("'") and p.endswith("'"):
                cleaned.append(p[1:-1])
            else:
                # try int
                if p.isdigit():
                    cleaned.append(int(p))
                else:
                    # try float
                    try:
                        fv = float(p)
                        cleaned.append(fv)
                    except:
                        # fallback string
                        cleaned.append(p)
        return cleaned

    def _build_row_data(self, table_obj, columns, vals):
        """
        Convert each val to the type declared in table_obj.schema, storing in a dict row_data.
        """
        schema_map = { name: ctype for (name, ctype) in table_obj.schema }
        row_data = {}

        for col, val in zip(columns, vals):
            if col not in schema_map:
                raise ValueError(f"Column '{col}' not in table schema.")
            col_type = schema_map[col]
            if col_type == "INT":
                val = int(val)
            elif col_type == "FLOAT":
                val = float(val)
            elif col_type == "DATETIME":
                # store as string for now
                if not isinstance(val, str):
                    val = str(val)
            else:
                # TEXT or unknown => str
                val = str(val)
            row_data[col] = val

        return row_data

    def _auto_fill_missing_cols(self, table_obj, row_data):
        """
        Fill missing columns with defaults if not supplied by user.
        E.g. DATETIME => current time, INT => int(time.time()), TEXT => "".
        """
        schema_map = { c[0]: c[1] for c in table_obj.schema }
        for col_name, col_type in schema_map.items():
            if col_name not in row_data:
                if col_type == "DATETIME":
                    row_data[col_name] = datetime.datetime.now().isoformat()
                elif col_type == "INT":
                    row_data[col_name] = int(time.time())
                elif col_type == "TEXT":
                    row_data[col_name] = ""

    def _must_embed(self, table_obj, row_data):
        """
        Option 2 approach: We require a text column (abstract or text).
        If none found => raise an error => user must supply text every time.
        """
        embed_col = None
        for col_name, col_type in table_obj.schema:
            # We'll treat "abstract" or "text" columns as embeddable
            if col_type == "TEXT" and col_name in ["abstract", "text"]:
                if col_name in row_data and row_data[col_name].strip():
                    embed_col = col_name
                    break

        if embed_col is None:
            raise ValueError("No text column (like 'abstract' or 'text') found with data. Insert disallowed if no text to embed.")

        text_val = row_data[embed_col]
        vector = self.embedding_model.encode(text_val).astype("float32")
        return vector

# --------------------------------------------------------------------------
# CLI if run directly
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("🧠 Multi-Table Vector SQL Processor (Always storing dict metadata, disallowing no-text inserts)")
    print("Supported Commands:\n")
    print("  CREATE TABLE tableName (col1 INT, col2 TEXT, col3 DATETIME, ...)")
    print("  SHOW TABLES;")
    print("  INSERT INTO tableName (text, metadata) VALUES ('some text', 'some meta');   # or typed columns")
    print("  SELECT ALL FROM tableName ORDER BY col [ASC|DESC];  (table-scan, sorted by col)")
    print("  SELECT * FROM tableName WHERE TEXT = 'search query' LIMIT N;  (semantic search)")
    print("\nType 'exit' to quit.\n")

    processor = QueryProcessor()

    while True:
        try:
            user_input = input("SQL> ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Exiting SQL processor.")
                break

            result = processor.execute_sql(user_input)
            print(result)
        except KeyboardInterrupt:
            print("\n👋 Exiting SQL processor.")
            break
        except Exception as e:
            print(f"⚠️ Runtime Error: {e}")
