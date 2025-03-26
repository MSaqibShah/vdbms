import re
import time
import datetime
import numpy as np
import csv
import io
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

        # 3) DROP TABLE
        elif sql_str.upper().startswith("DROP TABLE"):
            return self._handle_drop_table(sql_str)

        # 4) DELETE RECORD from table
        elif sql_str.upper().startswith("DELETE FROM"):
            return self._handle_delete_record(sql_str)

        # 5) SELECT ALL ... ORDER BY ...
        elif re.match(r"SELECT\s+ALL\s+FROM\s+\w+\s+ORDER\s+BY\s+\w+", sql_str, re.IGNORECASE):
            return self._handle_select_all_orderby(sql_str)

        # 6) SELECT * FROM <table name>; (full table-scan, no WHERE clause)
        elif re.match(r"SELECT\s+\*\s+FROM\s+\w+\s*;?\s*$", sql_str, re.IGNORECASE):
            return self._handle_select_all(sql_str)

        # 7) SELECT (semantic search)
        elif re.search(r"WHERE\s+TEXT\s*=\s*'.*?'(\s+LIMIT\s+\d+)?", sql_str, re.IGNORECASE):
            return self._handle_semantic_select(sql_str)

        # 8) INSERT typed or text
        elif re.match(r"INSERT\s+INTO\s+\w+\s*\(.*?\)\s*VALUES\s*\(.*?\)", sql_str, re.IGNORECASE):
            if re.search(r"\(\s*text\s*,\s*metadata\s*\)", sql_str, re.IGNORECASE):
                return self._handle_insert_text(sql_str)
            else:
                return self._handle_insert_typed(sql_str)
        
        # 9) UPDATE (semantic search)
        elif sql_str.upper().startswith("UPDATE"):
            return self._handle_update_record(sql_str)

        else:
            return "❌ Unsupported or invalid SQL statement."

    # ----------------------------------------------------------------------
    # 1) CREATE TABLE
    # ----------------------------------------------------------------------
    def _handle_create_table(self, sql):
        pattern = r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid CREATE TABLE statement."
        table_name = match.group(1)
        columns_str = match.group(2).strip()
        col_defs = [c.strip() for c in columns_str.split(',') if c.strip()]
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
        info = self.tables_manager.show_tables()
        if not info:
            return "⚠️ No tables found."
        headers = ["Table Name", "Table ID (sha256)", "Row Count", "Schema"]
        return tabulate(info, headers=headers, tablefmt="pretty")

    # ----------------------------------------------------------------------
    # 3) DROP TABLE
    # ----------------------------------------------------------------------
    def _handle_drop_table(self, sql):
        pattern = r"DROP\s+TABLE\s+(\w+)\s*;?\s*$"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid DROP TABLE syntax. Use: DROP TABLE table_name;"
        table_name = match.group(1)
        try:
            self.tables_manager.drop_table(table_name)
            return f"✅ Table '{table_name}' dropped."
        except ValueError as ve:
            return f"⚠️ {ve}"

    # ----------------------------------------------------------------------
    # 4) DELETE RECORD from table
    # ----------------------------------------------------------------------
    def _handle_delete_record(self, sql):
        """
        DELETE FROM <table_name> WHERE <column> = 'value';
        Deletes all records in the table where metadata[column] equals value.
        """
        pattern = r"DELETE\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*=\s*'(.*?)'\s*;?\s*$"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid DELETE syntax. Use: DELETE FROM table_name WHERE column = 'value';"
        table_name = match.group(1)
        column = match.group(2)
        value = match.group(3)
        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"
        # Call the delete_records method of the table (this must be implemented in vs.py)
        deleted_count = table_obj.delete_records(column, value)
        if deleted_count == 0:
            return f"⚠️ No records found in '{table_name}' where {column} = '{value}'."
        else:
            return f"✅ Deleted {deleted_count} record(s) from '{table_name}'."

    # ----------------------------------------------------------------------
    # 5) SELECT ALL ... ORDER BY ...
    # ----------------------------------------------------------------------
    def _handle_select_all_orderby(self, sql):
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
            row_str = f"VID={vid}, {meta}"
            rows.append([row_str])
        return tabulate(rows, headers=["Record"], tablefmt="pretty")

    # ----------------------------------------------------------------------
    # 6) SELECT * FROM <table name>;
    # ----------------------------------------------------------------------
    def _handle_select_all(self, sql):
        pattern = r"SELECT\s+\*\s+FROM\s+(\w+)\s*;?\s*$"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid SELECT * syntax. Use: SELECT * FROM table_name;"
        table_name = match.group(1)
        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"
        all_records = table_obj.list_all()
        if not all_records:
            return "⚠️ No records in this table."
        rows = []
        for (vid, meta) in all_records:
            row_str = f"VID={vid}, {meta}"
            rows.append([row_str])
        return tabulate(rows, headers=["Record"], tablefmt="pretty")

    # ----------------------------------------------------------------------
    # 7) SELECT (semantic search)
    # ----------------------------------------------------------------------
    def _handle_semantic_select(self, sql):
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
    # 8) INSERT TEXT (Legacy)
    # ----------------------------------------------------------------------
    def _handle_insert_text(self, sql):
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
        vector = self.embedding_model.encode(text_value).astype("float32")
        vectors = np.array([vector])
        metadata_dict = {
            "text": text_value,
            "user_metadata": user_meta_str
        }
        table_obj.insert_vectors(vectors, [metadata_dict])
        return f"✅ Inserted embedded vector from text '{text_value}' into table '{table_name}'"

    # ----------------------------------------------------------------------
    # 9) INSERT TYPED
    # ----------------------------------------------------------------------
    def _handle_insert_typed(self, sql):
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
        row_data = self._build_row_data(table_obj, columns, vals)
        self._auto_fill_missing_cols(table_obj, row_data)
        try:
            vector = self._must_embed(table_obj, row_data)
        except ValueError as ve:
            return f"❌ Insert disallowed: {ve}"
        vectors = np.array([vector], dtype="float32")
        table_obj.insert_vectors(vectors, [row_data])
        return f"✅ Inserted typed record into table '{table_name}' with columns {columns}"

    # ----------------------------------------------------------------------
    # 10) UPDATE RECORD from table
    # ----------------------------------------------------------------------
    def _handle_update_record(self, sql):
        """
        UPDATE <table_name> SET col1 = 'new_val', col2 = 'new_val2' WHERE condition_col = 'condition_value';
        """
        pattern = r"UPDATE\s+(\w+)\s+SET\s+(.*?)\s+WHERE\s+(\w+)\s*=\s*'(.*?)'\s*;?\s*$"
        match = re.match(pattern, sql, re.IGNORECASE)
        if not match:
            return "❌ Invalid UPDATE syntax. Expected: UPDATE <table> SET col1 = 'val1', col2 = 'val2' WHERE col = 'value';"
        table_name = match.group(1)
        set_clause = match.group(2)
        condition_column = match.group(3)
        condition_value = match.group(4)
        updates = {}
        assignments = set_clause.split(',')
        for assign in assignments:
            assign = assign.strip()
            m = re.match(r"(\w+)\s*=\s*'(.*?)'", assign)
            if not m:
                return f"❌ Invalid assignment in UPDATE: {assign}"
            col = m.group(1)
            new_val = m.group(2)
            updates[col] = new_val
        try:
            table_obj = self.tables_manager.get_table(table_name)
        except ValueError as ve:
            return f"⚠️ {ve}"
        updated_count = table_obj.update_records(condition_column, condition_value, updates)
        if updated_count == 0:
            return f"⚠️ No records updated in table '{table_name}'."
        else:
            return f"✅ Updated {updated_count} record(s) in table '{table_name}'."


    # ----------------------------------------------------------------------
    # HELPER METHODS
    # ----------------------------------------------------------------------
    def _parse_values(self, val_str):
        f = io.StringIO(val_str)
        reader = csv.reader(f, skipinitialspace=True)
        return next(reader)

    def _build_row_data(self, table_obj, columns, vals):
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
                if not isinstance(val, str):
                    val = str(val)
            else:
                val = str(val)
            row_data[col] = val
        return row_data

    def _auto_fill_missing_cols(self, table_obj, row_data):
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
        embed_col = None
        for col_name, col_type in table_obj.schema:
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
    print("  DROP TABLE tableName;")
    print("  DELETE FROM tableName WHERE column = 'value';  (delete record)")
    print("  INSERT INTO tableName (text, metadata) VALUES ('some text', 'some meta');   # or typed columns")
    print("  SELECT ALL FROM tableName ORDER BY col [ASC|DESC];  (table-scan, sorted by col)")
    print("  SELECT * FROM tableName;  (full table-scan)")
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
