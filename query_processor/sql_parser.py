# vector_db/query_processor/sql_parser.py
import sqlparse

def parse_sql(sql_query):
    parsed = sqlparse.parse(sql_query)[0]
    tokens = parsed.tokens

    table_name = None
    columns = []
    where_clause = None

    for token in tokens:
        if isinstance(token, sqlparse.sql.IdentifierList):
            for identifier in token.get_identifiers():
                columns.append(str(identifier))
        elif isinstance(token, sqlparse.sql.Identifier):
            if table_name is None:
                table_name = str(token)
        elif isinstance(token, sqlparse.sql.Where):
            where_clause = str(token)

    return {
        "table_name": table_name,
        "columns": columns,
        "where_clause": where_clause,
    }

# Example usage:
sql = "SELECT * FROM vectors WHERE similarity(vector, [0.1, 0.2, 0.3]) < 0.5;"
parsed_query = parse_sql(sql)
print(parsed_query)

# 2.  **Handling `similarity()`:**
#     * Identify the `similarity()` function in the `WHERE` clause.
#     * Extract the vector and threshold values.
#     * This will require custom logic, as `sqlparse` doesn't inherently understand your custom `similarity()` function.
# 3.  **Advanced Parsing:**
#     * Support more complex `WHERE` clauses (e.g., `AND`, `OR`, `NOT`).
#     * Handle different data types and operators.
#     * Handle `LIMIT` and `OFFSET` clauses.
# 4.  **Error Handling:**
#     * Implement robust error handling for invalid SQL syntax.