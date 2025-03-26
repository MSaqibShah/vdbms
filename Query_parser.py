import re
import csv
from io import StringIO
from tabulate import tabulate
import operator

SQL_KEYWORDS = {
    "CREATE", "DROP", "DATABASE", "USE", "TABLE", "SHOW",
    "INSERT", "INTO", "VALUES", "SELECT", "WHERE",
    "ORDER", "BY", "LIMIT", "UPDATE", "SET", "DELETE",
    "EMBEDDING", "DIMENSION", "AND", "OR", "NOT",
    "SLIKE", "ASC", "DESC", "FROM", "AS", "DATABASES", "TABLES", "FROM"
}
OPERATORS = {
    '=': operator.eq,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}


class QueryParser:

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.active_db = None  # Will hold a Database instance after `USE DATABASE`

    def execute(self, query: str):
        query = query.strip().rstrip(";")
        query = self._normalize_query_keywords(query)
        # upper_query = query.upper()
        print("===========================================")
        print(">> Query:", query)

        if query.startswith("CREATE DATABASE"):
            return self._create_database(query)
        elif query.startswith("DROP DATABASE"):
            return self._drop_database(query)
        elif query.startswith("USE DATABASE"):
            return self._use_database(query)
        elif query.startswith("SHOW DATABASES"):
            return self.show_databases()
        elif query.startswith("CREATE TABLE"):
            return self._create_table(query)
        elif query.startswith("DROP TABLE"):
            return self._drop_table(query)
        elif query == "SHOW TABLES":
            return self._show_tables()
        elif query.upper().startswith("INSERT INTO"):
            return self._insert_into(query)
        elif query.upper().startswith("SELECT"):
            return self._select(query)
        elif query.upper().startswith("UPDATE"):
            return self._update(query)
        elif query.upper().startswith("DELETE FROM"):
            return self._delete(query)
        else:
            raise ValueError(f"❌ Unsupported query: {query}")

    def _normalize_query_keywords(self, query: str) -> str:
        # Collapse common compound operators with optional whitespace
        query = re.sub(r'>\s*=', '>=', query)
        query = re.sub(r'<\s*=', '<=', query)
        query = re.sub(r'!\s*=', '!=', query)
        query = re.sub(r'=\s*=', '==', query)

        # Match words and punctuation as tokens
        tokens = re.findall(r'\w+|[(),;=*<>!]', query)

        # Reconstruct query with SQL keywords in uppercase
        normalized_tokens = [
            t.upper() if t.upper() in SQL_KEYWORDS else t
            for t in tokens
        ]

        # Rebuild the query with proper spacing
        result = ""
        for i, token in enumerate(normalized_tokens):
            if i == 0:
                result += token
            elif token in {",", ";", ")", "*", "=", "<", ">", "!=", "<=", ">=", "=="}:
                result += " " + token
            elif normalized_tokens[i - 1] in {"(", "="}:
                result += token
            else:
                result += " " + token

        result = re.sub(r'>\s*=', '>=', result)
        result = re.sub(r'<\s*=', '<=', result)
        result = re.sub(r'!\s*=', '!=', result)
        result = re.sub(r'=\s*=', '==', result)

        return result

    def _create_database(self, query):
        tokens = query.split()
        if len(tokens) != 3:
            raise ValueError("Syntax error in CREATE DATABASE statement.")
        db_name = tokens[2]
        self.db_manager.create_database(db_name)
        return f"✅ Database '{db_name}' created."

    def _drop_database(self, query):
        tokens = query.split()
        if len(tokens) != 3:
            raise ValueError("Syntax error in DROP DATABASE statement.")
        db_name = tokens[2]
        self.db_manager.drop_database(db_name)
        return f"🗑️ Database '{db_name}' deleted."

    def _use_database(self, query):
        tokens = query.split()
        if len(tokens) != 3:
            raise ValueError("Syntax error in USE DATABASE statement.")
        db_name = tokens[2]
        self.active_db = self.db_manager.use_database(db_name)
        return f"📂 Switched to database '{db_name}'."

    def show_databases(self):
        databases = self.db_manager.list_databases()
        if not databases:
            return "No databases found."
        return "\n".join([f"📁 {db}" for db in databases])

    def _create_table(self, query):
        if not self.active_db:
            raise Exception(
                "❌ No active database. Use 'USE DATABASE dbname;' first.")

        try:
            # Extract the part between parentheses (schema definition)
            before_paren, after_paren = query.split("(", 1)
            column_block, rest = after_paren.split(")", 1)

            # Parse table name (3rd token after "CREATE TABLE")
            tokens = before_paren.strip().split()
            if len(tokens) < 3 or tokens[0].upper() != "CREATE" or tokens[1].upper() != "TABLE":
                raise Exception("Invalid CREATE TABLE syntax.")
            table_name = tokens[2]

            # Parse schema: a list of "col_name TYPE" pairs
            schema = {}
            for col in column_block.split(","):
                parts = col.strip().split()
                if len(parts) != 2:
                    raise Exception(f"Invalid column definition: '{col}'")
                col_name, col_type = parts
                schema[col_name.strip()] = col_type.strip()

            # Use regex to extract EMBEDDING and DIMENSION values
            embedding_match = re.search(
                r'EMBEDDING\s*\(\s*(\w+)\s*\)', rest, re.IGNORECASE)
            dimension_match = re.search(
                r'DIMENSION\s+(\d+)', rest, re.IGNORECASE)

            if not embedding_match or not dimension_match:
                raise Exception("Missing EMBEDDING column or DIMENSION.")

            embedding_column = embedding_match.group(1)
            dimension = int(dimension_match.group(1))

            self.active_db.create_table(
                table_name=table_name,
                dimension=dimension,
                schema=schema,
                embedding_column=embedding_column
            )

            return f"✅ Table '{table_name}' created."

        except Exception as e:
            return f"❌ Error parsing CREATE TABLE: {e}"

    def _drop_table(self, query):
        if not self.active_db:
            raise Exception("❌ No active database.")

        tokens = query.strip().split()
        if len(tokens) != 3:
            raise Exception("Syntax error in DROP TABLE.")
        table_name = tokens[2]

        self.active_db.drop_table(table_name)
        return f"🗑️ Table '{table_name}' dropped."

    def _show_tables(self):
        if not self.active_db:
            raise Exception("❌ No active database.")
        tables = self.active_db.list_tables()
        return "\n".join(
            [f"📄 {t['table_name']} (columns: {list(t['schema'].keys())})" for t in tables]
        ) or "No tables found."

    def _insert_into(self, query):
        if not self.active_db:
            raise Exception("❌ No active database selected.")

        try:
            match = re.match(
                r"INSERT\s+INTO\s+(\w+)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
                query, re.IGNORECASE
            )
            if not match:
                raise Exception("Invalid INSERT syntax.")

            table_name = match.group(1)
            columns_str = match.group(2)
            values_str = match.group(3)

            columns = [c.strip() for c in columns_str.split(",")]

            reader = csv.reader(StringIO(values_str), skipinitialspace=True)
            raw_values = next(reader)

            # Optional: Convert numbers from strings to proper types
            values = [
                int(v) if v.isdigit()
                else float(v) if v.replace('.', '', 1).isdigit()
                else v
                for v in raw_values
            ]

            if len(columns) != len(values):
                raise Exception("Mismatch between columns and values.")

            record = dict(zip(columns, values))
            table = self.active_db.get_table(table_name)
            vector_id = table.insert(record)

            return f"✅ Inserted into '{table_name}' with vector ID {vector_id}"

        except Exception as e:
            return f"❌ Error parsing INSERT INTO: {e}"

    def _select(self, query):
        if not self.active_db:
            raise Exception("❌ No active database selected.")

        try:
            # Remove trailing semicolon
            query = query.rstrip(";")

            # Parse the FROM clause to get the table name
            match_from = re.search(
                r"SELECT\s+\*\s+FROM\s+(\w+)", query, re.IGNORECASE)
            if not match_from:
                raise Exception("Invalid SELECT syntax.")
            table_name = match_from.group(1)

            # Initialize filter and semantic query components
            # List of (key, operator, value) tuples for traditional conditions
            filter_by = []
            semantic_query_parts = []  # List of semantic search strings
            sort_by = None
            ascending = True
            limit = None

            # Get table and its schema info
            table = self.active_db.get_table(table_name)
            # Traditional schema (e.g., {'title': 'TEXT', ...})
            schema = table.get_schema()
            embedding_column = table.get_embedding_column()  # The embedding column name

            # Parse the WHERE clause (if present)
            where_match = re.search(
                r"WHERE\s+(.*?)(\s+ORDER BY|\s+LIMIT|$)", query, re.IGNORECASE)
            if where_match:
                where_clause = where_match.group(1).strip()
                conditions = [cond.strip()
                              for cond in where_clause.split("AND")]
                for cond in conditions:
                    # Check if the condition uses SLIKE
                    if re.search(r"\bSLIKE\b", cond, re.IGNORECASE):
                        # Use regex to extract the column and the search string
                        m = re.match(r"(\w+)\s+SLIKE\s+(.*)",
                                     cond, re.IGNORECASE)
                        if m:
                            col = m.group(1).strip()
                            val = m.group(2).strip().strip('"').strip("'")
                            # Ensure SLIKE is used only on the embedding column
                            if col.lower() != embedding_column.lower():
                                raise Exception(
                                    f"SLIKE operator can only be used on the embedding column '{embedding_column}'.")
                            semantic_query_parts.append(val)
                        else:
                            raise Exception("Invalid SLIKE syntax.")
                    else:
                        # Traditional condition: use your existing _parse_condition() helper.
                        filter_by.append(self._parse_condition(cond, schema))

            # Combine semantic query parts (if any) into a single query_text
            query_text = " ".join(
                semantic_query_parts) if semantic_query_parts else None

            # Parse ORDER BY
            order_match = re.search(
                r"ORDER\s+BY\s+(\w+)(\s+ASC|\s+DESC)?", query, re.IGNORECASE)
            if order_match:
                sort_by = order_match.group(1)
                direction = order_match.group(2)
                if direction and direction.strip().upper() == "DESC":
                    ascending = False

            # Parse LIMIT
            limit_match = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
            if limit_match:
                limit = int(limit_match.group(1))

            # Execute SELECT with both traditional filters and semantic query_text
            results = table.select(
                filter_by=filter_by if filter_by else None,
                sort_by=sort_by,
                ascending=ascending,
                query_text=query_text,
                limit=limit
            )
            return self._format_select_results(results)

        except Exception as e:
            return f"❌ Error parsing SELECT: {e}"

    def _format_select_results(self, results):
        if not results:
            return "⚠️ No results found."

        table_data = []
        headers = ["Vector ID"]  # Default first column

        for row in results:
            if len(row) == 2:
                vector_id, metadata = row
                if not headers or len(headers) == 1:
                    headers = ["Vector ID"] + list(metadata.keys())
                table_data.append([vector_id] + list(metadata.values()))
            elif len(row) == 3:
                vector_id, distance, metadata = row
                if not headers or len(headers) == 1:
                    headers = ["Vector ID", "Distance"] + list(metadata.keys())
                table_data.append(
                    [vector_id, round(distance, 4)] + list(metadata.values()))

        return tabulate(table_data, headers=headers, tablefmt="fancy_grid")

    def _parse_condition(self, cond, schema):
        cond = cond.strip()
        for op in sorted(OPERATORS, key=lambda x: -len(x)):  # Check longest ops first
            if op in cond:
                key, val = map(str.strip, cond.split(op, 1))
                dtype = schema.get(key).lower()

                if not dtype:
                    raise Exception(f"Column '{key}' not found in schema.")

                if dtype == "number":
                    val = float(val) if "." in val else int(val)
                elif dtype == "text":
                    val = val.strip().strip('"').strip("'")
                else:
                    raise Exception(f"Unsupported data type '{dtype}'")

                return (key, OPERATORS[op], val)
        raise Exception("Invalid condition syntax")

    def _update(self, query):
        if not self.active_db:
            raise Exception("❌ No active database selected.")
        try:
            # Remove trailing semicolon
            query = query.rstrip(";")

            # Use regex to capture table name, SET clause, and optional WHERE clause.
            # The LIMIT clause (if any) will be handled separately.
            pattern = r"UPDATE\s+(\w+)\s+SET\s+(.*?)(?:\s+WHERE\s+(.*?))?(?:\s+LIMIT\s+(\d+))?$"
            match = re.match(pattern, query, re.IGNORECASE)
            if not match:
                raise Exception(
                    "Invalid UPDATE syntax. Ensure a SET clause is present.")

            table_name = match.group(1)
            set_clause = match.group(2)
            where_clause = match.group(3)  # may be None
            limit = int(match.group(4)) if match.group(4) else None

            # Parse the SET clause into a dictionary of updates.
            updates = {}
            for assignment in set_clause.split(","):
                parts = assignment.split("=")
                if len(parts) != 2:
                    raise Exception(
                        f"Invalid assignment in SET clause: '{assignment}'")
                key = parts[0].strip()
                val = parts[1].strip().strip('"').strip("'")
                # You can do type conversion later (or in Table.update) based on schema.
                updates[key] = val

            # Parse the WHERE clause, if present.
            # List of (key, operator_func, value) for traditional conditions
            filter_by = []
            # List of semantic search strings (from SLIKE conditions)
            semantic_query_parts = []
            table = self.active_db.get_table(table_name)
            schema = table.get_schema()
            embedding_column = table.get_embedding_column()

            if where_clause and where_clause.strip():
                conditions = [cond.strip()
                              for cond in where_clause.split("AND")]
                for cond in conditions:
                    # Check for SLIKE (case-insensitive)
                    if re.search(r"\bSLIKE\b", cond, re.IGNORECASE):
                        m = re.match(r"(\w+)\s+SLIKE\s+(.*)",
                                     cond, re.IGNORECASE)
                        if m:
                            col = m.group(1).strip()
                            val = m.group(2).strip().strip('"').strip("'")
                            # Ensure SLIKE is applied only on the embedding column.
                            if col.lower() != embedding_column.lower():
                                raise Exception(
                                    f"SLIKE operator can only be used on the embedding column '{embedding_column}'.")
                            semantic_query_parts.append(val)
                        else:
                            raise Exception("Invalid SLIKE syntax.")
                    else:
                        # Traditional condition: parse it using your existing _parse_condition().
                        parsed_condition = self._parse_condition(cond, schema)
                        filter_by.append(parsed_condition)

                # Combine any SLIKE conditions into one semantic search query_text.
                query_text = " ".join(
                    semantic_query_parts) if semantic_query_parts else None

                # Execute the update: pass both traditional filters and semantic query_text.
                updated_count = table.update(
                    updated_fields=updates,
                    filter_by=filter_by if filter_by else None,
                    query_text=query_text,
                    limit=limit
                )

                return f"✅ Updated {updated_count} record(s) in '{table_name}'."

        except Exception as e:
            return f"❌ Error parsing UPDATE: {e}"

    def _delete(self, query):
        if not self.active_db:
            raise Exception("❌ No active database selected.")
        try:
            # Remove trailing semicolon
            query = query.rstrip(";")

            # Robust DELETE regex: supports optional WHERE, ORDER BY, and LIMIT clauses.
            pattern = (r"DELETE\s+FROM\s+(\w+)"
                       r"(?:\s+WHERE\s+(.*?))?"
                       r"(?:\s+ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?)?"
                       r"(?:\s+LIMIT\s+(\d+))?$")
            match = re.match(pattern, query, re.IGNORECASE)
            if not match:
                raise Exception("Invalid DELETE syntax.")

            table_name = match.group(1)
            where_clause = match.group(2)  # may be None
            sort_by = match.group(3)       # may be None
            order = match.group(4)         # may be None
            limit = int(match.group(5)) if match.group(5) else None

            ascending = True if not order or order.upper() == "ASC" else False

            # Initialize filter components
            # List of traditional filter conditions (tuples)
            filter_by = []
            semantic_query_parts = []  # List of semantic search strings from SLIKE conditions

            # Get table and schema info
            table = self.active_db.get_table(table_name)
            schema = table.get_schema()
            embedding_column = table.get_embedding_column()

            # Parse WHERE clause if present and non-empty.
            if where_clause and where_clause.strip():
                conditions = [cond.strip()
                              for cond in where_clause.split("AND")]
                for cond in conditions:
                    # Check for SLIKE operator (case-insensitive)
                    if re.search(r"\bSLIKE\b", cond, re.IGNORECASE):
                        m = re.match(r"(\w+)\s+SLIKE\s+(.*)",
                                     cond, re.IGNORECASE)
                        if m:
                            col = m.group(1).strip()
                            val = m.group(2).strip().strip('"').strip("'")
                            # Enforce that SLIKE is used only on the embedding column.
                            if col.lower() != embedding_column.lower():
                                raise Exception(
                                    f"SLIKE operator can only be used on the embedding column '{embedding_column}'.")
                            semantic_query_parts.append(val)
                        else:
                            raise Exception("Invalid SLIKE syntax in DELETE.")
                    else:
                        # Traditional condition: use existing _parse_condition() helper.
                        filter_by.append(self._parse_condition(cond, schema))

            # Combine any SLIKE conditions into a single semantic query string.
            query_text = " ".join(
                semantic_query_parts) if semantic_query_parts else None

            # Execute delete: pass both traditional filters and semantic query_text, plus sorting/limit.
            deleted_count = table.delete(
                filter_by=filter_by if filter_by else None,
                sort_by=sort_by,
                ascending=ascending,
                query_text=query_text,
                limit=limit
            )

            return f"🗑️ Deleted {deleted_count} record(s) from '{table_name}'."

        except Exception as e:
            return f"❌ Error parsing DELETE: {e}"
