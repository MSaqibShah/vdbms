import numpy as np
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import os
import operator
from flask_cors import CORS  # <-- Import flask-cors

# Import your DBMS components. Adjust the imports based on your project structure.
from Table import Table, TableManager
from Query_parser import QueryParser
from Databse import Database, DatabaseManager  # adjust accordingly
from flask import url_for

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# Global DB manager and query parser instances.
db_manager = DatabaseManager()
query_parser = QueryParser(db_manager)


def convert_numpy_types(obj):
    """
    Recursively converts numpy data types to native Python types.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


OPERATORS = {
    '=': operator.eq,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge
}

# -------------------- SWAGGER SPECIFICATION --------------------

SWAGGER_URL = '/docs'
API_URL = '/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Custom DBMS API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


@app.route('/swagger.json')
def swagger_spec():
    spec = {
        "swagger": "2.0",
        "info": {
            "title": "Custom DBMS API",
            "description": "API for managing databases, tables, and records in a custom DBMS.",
            "version": "1.0.0"
        },
        "basePath": "/",
        "schemes": ["http"],
        "paths": {}
    }

    def param_schema(schema, example=None):
        if example:
            schema["example"] = example  # Embed example directly into schema
        return [{
            "in": "body",
            "name": "body",
            "required": True,
            "schema": schema
        }]

    def endpoint(summary, desc, method, parameters=None, example=None):
        return {
            method: {
                "summary": summary,
                "description": desc,
                "consumes": ["application/json"],
                "parameters": param_schema(parameters, example) if parameters else [],
                "responses": {
                    "200": {"description": f"{summary} success"},
                    "400": {"description": f"{summary} error"}
                }
            }
        }

    def schema_props(required, props):
        return {
            "type": "object",
            "properties": props,
            "required": required
        }

    spec["paths"] = {
        "/database": {
            **endpoint("Create Database", "Creates a new database.", "post",
                       schema_props(
                           ["db_name"], {"db_name": {"type": "string"}}),
                       {"db_name": "analytics"}),
            **endpoint("Drop Database", "Drops an existing database.", "delete",
                       schema_props(
                           ["db_name"], {"db_name": {"type": "string"}}),
                       {"db_name": "analytics"})
        },
        "/databases": {
            "get": {
                "summary": "List all databases",
                "description": "Lists all databases.",
                "responses": {
                    "200": {"description": "Success"},
                    "400": {"description": "Error"}
                }
            }
        },
        "/database/use": endpoint("Use Database", "Sets the active database.", "post",
                                  schema_props(
                                      ["db_name"], {"db_name": {"type": "string"}}),
                                  {"db_name": "analytics"}),
        "/table": {
            **endpoint("Create Table", "Creates a new table.", "post",
                       schema_props(["table_name", "schema", "embedding_column", "dimension"], {
                           "table_name": {"type": "string"},
                           "schema": {"type": "object"},
                           "embedding_column": {"type": "string"},
                           "dimension": {"type": "integer"}
                       }),
                       {
                           "table_name": "products",
                           "schema": {
                               "title": "TEXT",
                               "description": "TEXT",
                               "price": "NUMBER"
                           },
                           "embedding_column": "description",
                           "dimension": 384
                       }),
            **endpoint("Drop Table", "Drops a table.", "delete",
                       schema_props(["table_name"], {
                                    "table_name": {"type": "string"}}),
                       {"table_name": "products"})
        },
        "/tables": {
            "get": {
                "summary": "List all tables",
                "description": "Lists all tables in the active database.",
                "responses": {
                    "200": {"description": "Success"},
                    "400": {"description": "Error"}
                }
            }
        },
        "/table/insert": endpoint("Insert Record", "Insert a record into a table.", "post",
                                  schema_props(["table_name", "record"], {
                                      "table_name": {"type": "string"},
                                      "record": {"type": "object"}
                                  }),
                                  {
                                      "table_name": "products",
                                      "record": {
                                          "title": "USB Hub",
                                          "description": "Multiport hub",
                                          "price": 899
                                      }
                                  }),
        "/table/select": endpoint("Select Records", "Selects records from a table.", "post",
                                  schema_props(["table_name"], {
                                      "table_name": {"type": "string"},
                                      "filter_by": {"type": "object"},
                                      "order_by": {"type": "string"},
                                      "direction": {"type": "string"},
                                      "limit": {"type": "integer"},
                                      "query_text": {"type": "string"}
                                  }),
                                  {
                                      "table_name": "products",
                                      "filter_by": {
                                          "column": "price",
                                          "operator": ">",
                                          "value": 1000
                                      },
                                      "order_by": "price",
                                      "direction": "DESC",
                                      "limit": 10
                                  }),
        "/table/update": endpoint("Update Records", "Updates records in a table.", "put",
                                  schema_props(["table_name", "updates"], {
                                      "table_name": {"type": "string"},
                                      "updates": {"type": "object"},
                                      "filter_by": {"type": "object"},
                                      "order_by": {"type": "string"},
                                      "direction": {"type": "string"},
                                      "limit": {"type": "integer"},
                                      "query_text": {"type": "string"}
                                  }),
                                  {
                                      "table_name": "products",
                                      "updates": {"price": 2999},
                                      "filter_by": {
                                          "column": "price",
                                          "operator": "=",
                                          "value": 899
                                      },
                                      "limit": 5
                                  }),
        "/table/delete": endpoint("Delete Records", "Deletes records from a table.", "delete",
                                  schema_props(["table_name"], {
                                      "table_name": {"type": "string"},
                                      "filter_by": {"type": "object"},
                                      "order_by": {"type": "string"},
                                      "direction": {"type": "string"},
                                      "limit": {"type": "integer"},
                                      "query_text": {"type": "string"}
                                  }),
                                  {
                                      "table_name": "products",
                                      "filter_by": {
                                          "column": "price",
                                          "operator": ">",
                                          "value": 1000
                                      },
                                      "limit": 2
                                  }),
        "/sql": endpoint("Execute SQL", "Executes a raw SQL query.", "post",
                         schema_props(["query"], {
                             "query": {"type": "string"}
                         }),
                         {"query": "SELECT * FROM products WHERE price > 1000;"})

    }
    return jsonify(spec)

# -------------------- DATABASE ROUTES --------------------


@app.route('/database', methods=['POST'])
def create_database():
    """
    Create a new database.
    Expects JSON: { "db_name": "<database_name>" }
    """
    data = request.get_json()
    db_name = data.get('db_name')
    if not db_name:
        return jsonify({'error': "Missing 'db_name' parameter."}), 400
    try:
        db_manager.create_database(db_name)
        return jsonify({'message': f"Database '{db_name}' created."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/database', methods=['DELETE'])
def drop_database():
    """
    Drop an existing database.
    Expects JSON: { "db_name": "<database_name>" }
    """
    data = request.get_json()
    db_name = data.get('db_name')
    if not db_name:
        return jsonify({'error': "Missing 'db_name' parameter."}), 400
    try:
        db_manager.drop_database(db_name)
        return jsonify({'message': f"Database '{db_name}' deleted."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/databases', methods=['GET'])
def list_databases():
    """
    List all available databases.
    """
    try:
        databases = db_manager.list_databases()
        return jsonify({'databases': databases})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/database/use', methods=['POST'])
def use_database():
    """
    Set the active database.
    Expects JSON: { "db_name": "<database_name>" }
    """
    data = request.get_json()
    db_name = data.get('db_name')
    if not db_name:
        return jsonify({'error': "Missing 'db_name' parameter."}), 400
    try:
        database = db_manager.use_database(db_name)
        # Update the QueryParser's active database
        query_parser.active_db = database
        return jsonify({'message': f"Switched to database '{db_name}'."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# -------------------- TABLE ROUTES --------------------


@app.route('/table', methods=['POST'])
def create_table():
    """
    Create a new table in the active database.
    Expects JSON with:
      - table_name: string
      - schema: dictionary of column definitions, e.g., {"title": "TEXT", "price": "NUMBER"}
      - embedding_column: string (column name for embedding)
      - dimension: integer (dimension value)
    """
    data = request.get_json()
    table_name = data.get('table_name')
    schema = data.get('schema')
    embedding_column = data.get('embedding_column')
    dimension = data.get('dimension')

    if table_name is None or schema is None or embedding_column is None or dimension is None:
        return jsonify({'error': "Missing one or more required parameters: 'table_name', 'schema', 'embedding_column', 'dimension'."}), 400

    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400
    try:
        query_parser.active_db.create_table(
            table_name=table_name,
            schema=schema,
            embedding_column=embedding_column,
            dimension=dimension
        )
        return jsonify({'message': f"Table '{table_name}' created."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/table', methods=['DELETE'])
def drop_table():
    """
    Drop a table from the active database.
    Expects JSON: { "table_name": "<table_name>" }
    """
    data = request.get_json()
    table_name = data.get('table_name')
    if not table_name:
        return jsonify({'error': "Missing 'table_name' parameter."}), 400
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400
    try:
        query_parser.active_db.drop_table(table_name)
        return jsonify({'message': f"Table '{table_name}' dropped."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/tables', methods=['GET'])
def list_tables():
    """
    List all tables in the active database.
    """
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400
    try:
        tables = query_parser.active_db.list_tables()
        return jsonify({'tables': tables})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# -------------------- RECORD ROUTES --------------------


@app.route('/table/insert', methods=['POST'])
def insert_into_table():
    """
    Insert a record into a table.
    Expects JSON with:
      - table_name: string
      - record: a dictionary of column values, e.g.,
          { "title": "USB Hub", "description": "Multiport hub", "price": 899 }
    """
    data = request.get_json()
    table_name = data.get('table_name')
    record = data.get('record')
    if not table_name or not record:
        return jsonify({'error': "Missing 'table_name' or 'record' parameter."}), 400
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400
    try:
        table = query_parser.active_db.get_table(table_name)
        vector_id = table.insert(record)
        return jsonify({'message': f"Inserted into '{table_name}' with vector ID {vector_id}."})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/table/select', methods=['POST'])
def select_from_table():
    """
    Select records from a table.
    Expects JSON with:
      - table_name: string
      - Optional parameters:
          - where: string condition (e.g., "price > 1000")
          - order_by: column name to sort by
          - direction: "ASC" or "DESC" (default is ASC)
          - limit: integer limit on number of records
    Constructs a SQL-like query string and calls the QueryParser.
    """
    data = request.get_json()
    table_name = data.get('table_name')
    if not table_name:
        return jsonify({'error': "Missing 'table_name' parameter."}), 400
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400

    try:
        db = query_parser.active_db
        table = db.get_table(table_name)
        if not table:
            return jsonify({'error': f"Table '{table_name}' does not exist."}), 400

        filter = data.get('filter_by')

        filter["operator"] = OPERATORS[filter["operator"]]

        filter_by = (filter['column'], filter["operator"], filter["value"])
        result = table.select(
            filter_by=[filter_by],
            sort_by=data.get('order_by'),
            ascending=False if data.get('direction') == 'DESC' else True,
            limit=data.get('limit'),
            query_text=data.get('query_text')
        )

        response = []

        for data in result:
            if len(data) == 2:
                obj = {
                    "vector_id": data[0],
                    "columns": data[1]
                }
                response.append(obj)
            else:
                obj = {
                    "vector_id": convert_numpy_types(data[0]),
                    "search_distance": convert_numpy_types(data[1]),
                    "columns": data[2]
                }
                response.append(obj)

        return jsonify({'result': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/table/update', methods=['PUT'])
def update_table():
    """
    Update records in a table.
    Expects JSON with:
      - table_name: string
      - updates: a dictionary of column updates, e.g., { "price": 2999 }
      - Optional:
          - where: string condition (e.g., "price = 899")
          - limit: integer
    Constructs an UPDATE SQL-like query string and executes it.
    """
    data = request.get_json()
    table_name = data.get('table_name')
    updates = data.get('updates')
    if not table_name:
        return jsonify({'error': "Missing 'table_name' parameter."}), 400
    if not updates:
        return jsonify({'error': "Missing 'updates' parameter."}), 400

    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400

    try:
        db_manager = query_parser.active_db
        table = db_manager.get_table(table_name)
        if not table:
            return jsonify({'error': f"Table '{table_name}' does not exist."}), 400

        filter = data.get('filter_by')

        filter["operator"] = OPERATORS[filter["operator"]]

        filter_by = (filter['column'], filter["operator"], filter["value"])
        result = table.update(
            updated_fields=updates,
            filter_by=[filter_by],
            sort_by=data.get('order_by'),
            ascending=False if data.get('direction') == 'DESC' else True,
            limit=data.get('limit'),
            query_text=data.get('query_text')
        )

        return jsonify({'result': {"updatedCount": result, "table_name": table_name}}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/table/delete', methods=['DELETE'])
def delete_from_table():
    """
    Delete records from a table.
    Expects JSON with:
      - table_name: string
      - Optional parameters:
          - filter_by: {
              "column": "price",
              "operator": ">",
              "value": 100
            }
          - query_text: string (for semantic search on embedding column)
          - order_by: string (column name)
          - direction: "ASC" or "DESC"
          - limit: integer
    """
    data = request.get_json()
    table_name = data.get('table_name')

    if not table_name:
        return jsonify({'error': "Missing 'table_name' parameter."}), 400
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400

    try:
        db = query_parser.active_db
        table = db.get_table(table_name)
        if not table:
            return jsonify({'error': f"Table '{table_name}' does not exist."}), 400

        # Build filter_by
        filter = data.get('filter_by')
        filter_by = None
        if filter:
            op = OPERATORS.get(filter.get('operator'))
            if not op:
                return jsonify({'error': "Invalid operator in 'filter_by'."}), 400
            filter_by = [(filter['column'], op, filter['value'])]

        # Perform deletion
        result = table.delete(
            filter_by=filter_by,
            query_text=data.get('query_text'),
            sort_by=data.get('order_by'),
            ascending=False if data.get('direction') == 'DESC' else True,
            limit=data.get('limit')
        )

        return jsonify({'result': {"deletedCount": result, table_name: table_name}}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/sql', methods=['POST'])
def execute_sql():
    """
    Execute a raw SQL query.
    Expects JSON: { "query": "<SQL_query>" }
    """
    data = request.get_json()
    sql_query = data.get('query')
    if not sql_query:
        return jsonify({'error': "Missing 'query' parameter."}), 400
    if not query_parser.active_db:
        return jsonify({'error': "No active database. Use '/database/use' first."}), 400

    try:
        result = query_parser.execute(sql_query, json_resp=True)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5010)
