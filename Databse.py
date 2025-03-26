import os
import shutil

from Table import Table
from Table import TableManager
import os

from Query_parser import QueryParser


class Database:
    def __init__(self, name, root_path="./store"):
        self.name = name
        self.path = os.path.join(root_path, name)
        os.makedirs(self.path, exist_ok=True)

        self.table_manager = TableManager(database_path=self.path)

    def list_tables(self):
        return self.table_manager.list_tables()

    def create_table(self, *args, **kwargs):
        self.table_manager.create_table(*args, **kwargs)

        # Try to extract table_name from kwargs or args
        table_name = kwargs.get("table_name") or (args[0] if args else None)

        if not table_name:
            raise ValueError(
                "Missing table_name for table retrieval after creation.")

        return self.get_table(table_name)

    def drop_table(self, table_name):
        return self.table_manager.drop_table(table_name)

    def get_table(self, table_name):
        return self.table_manager.get_table(table_name)


class DatabaseManager:
    def __init__(self, root_path="./store"):
        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)
        self.current_db = None

    def create_database(self, db_name):
        db_path = os.path.join(self.root_path, db_name)
        if os.path.exists(db_path):
            raise Exception(f"Database '{db_name}' already exists.")
        os.makedirs(db_path)
        print(f"✅ Database '{db_name}' created.")

    def drop_database(self, db_name):
        db_path = os.path.join(self.root_path, db_name)
        if not os.path.exists(db_path):
            raise Exception(f"Database '{db_name}' does not exist.")

        if self.current_db == db_name:
            raise Exception(
                f"Cannot delete the current database '{db_name}'. Switch to another database first.")

        shutil.rmtree(db_path)
        print(f"🗑️ Database '{db_name}' deleted.")

    def list_databases(self):
        return [
            name for name in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, name))
        ]

    def use_database(self, db_name):
        db_path = os.path.join(self.root_path, db_name)
        if not os.path.exists(db_path):
            raise Exception(f"Database '{db_name}' does not exist.")
        self.current_db = db_name
        return Database(name=db_name, root_path=self.root_path)


if __name__ == "__main__":
    dbm = DatabaseManager()
    parser = QueryParser(dbm)

    # print(parser.execute("CREATE DATABASE analytics;"))
    # print(parser.execute("USE DATABASE analytics;"))
    # print(parser.execute("SHOW DATABASES;"))
    # print(parser.execute(
    #     "CREATE TABLE products (title TEXT, description TEXT, price NUMBER) EMBEDDING(description) DIMENSION 384;"))
    # print(parser.execute(
    #     'INSERT INTO products (title, description, price) VALUES ("USB Hub", "Multiport hub for USB-C ", 899);'))
    # print(parser.execute(
    #     'INSERT INTO products (title, description, price) VALUES ("USB Hub 2", "Multiport hub for USB-C  2", 999);'))
    # print(parser.execute(
    #     'INSERT INTO products (title, description, price) VALUES ("USB Hub 3", "Multiport hub for USB-C devices 3", 1099);'))
    # print(parser.execute(
    #     'INSERT INTO products (title, description, price) VALUES ("USB Hub 4", "Multiport hub for USB-C devices 4", 1199);'))
    # print(parser.execute("SELECT * FROM products;"))
    # print(parser.execute("SELECT * FROM products ORDER BY price DESC;"))
    # print(parser.execute("SELECT * FROM products ORDER BY price DESC LIMIT 2;"))
    # print(parser.execute("SELECT * FROM products WHERE price != 999 LIMIT 2;"))
    # print(parser.execute(
    #     "SELECT * FROM products WHERE price > 1099 ORDER BY price DESC LIMIT 2;"))
    # parser.execute("SHOW TABLES;")
    # print(parser.execute("UPDATE products SET price = 2999 where price = 899;"))
    # print(parser.execute("UPDATE products SET price = 2999 where price = 999;"))
    # print(parser.execute("SELECT * FROM products;"))
    # print(parser.execute("UPDATE products SET price = 999999 where price = 2999;"))
    # print(parser.execute("SELECT * FROM products;"))
    # print(parser.execute("UPDATE products SET price = 1"))
    # print(parser.execute("SELECT * FROM products;"))
    # print(parser.execute("DELETE FROM products WHERE price = 899;"))
    # print(parser.execute(
    # "SELECT * FROM products where price=1099 AND description SLIKE devices 3 LIMIT 2;"))
    # print(parser.execute(
    # "UPDATE products SET price=20190 where price=1099 AND description SLIKE devices 3 LIMIT 2;"))
    # print(parser.execute("SELECT * FROM products;"))

    # print(parser.execute(
    #     "DELETE FROM products where description SLIKE devices 3 LIMIT 1;"))
    # print(parser.execute("SELECT * FROM products;"))
    # parser.execute("DROP TABLE products;")
    # parser.execute("USE DATABASE shopdb;")
    # print(parser.execute("DROP DATABASE analytics;"))

    while True:
        try:
            query = input("SQL> ").strip()
            if query.lower() == "exit":
                print("Exiting...")
                break
            result = parser.execute(query)
            print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
