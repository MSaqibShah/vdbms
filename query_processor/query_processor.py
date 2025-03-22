import re
import numpy as np
from VectorSearch import VectorSearch

class QueryProcessor:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def process_query(self, query):
        query = query.strip()

        try:
            if query.upper().startswith("SELECT"):
                return self._handle_select(query)
            elif query.upper().startswith("INSERT"):
                return self._handle_insert(query)
            elif query.upper().startswith("UPDATE"):
                return self._handle_update(query)
            elif query.upper().startswith("DELETE"):
                return self._handle_delete(query)
            else:
                raise ValueError(f"Unsupported query type: {query.split()[0]}")
        except Exception as e:
            raise ValueError(f"Query processing failed: {e}")

    def _parse_vector(self, vector_string):
        try:
            vector = [float(x) for x in vector_string.split(",")]
            return np.array(vector, dtype='float32')
        except ValueError:
            raise ValueError(f"Invalid vector format: {vector_string}")

    def _handle_select(self, query):
        pattern = re.compile(
            r"SELECT\s+\*\s+FROM\s+VECTORS"
            r"(?:\s+WHERE\s+VECTOR\s*=\s*\[(.*?)\])?"
            r"(?:\s+AND\s+METADATA\s+LIKE\s+'(.*?)')?"
            r"(?:\s+ORDER\s+BY\s+([a-zA-Z0-9_]+)\s+(ASC|DESC))?"
            r"(?:\s+LIMIT\s+(\d+))?"
            r"(?:\s+OFFSET\s+(\d+))?",
            re.IGNORECASE
        )

        match = pattern.match(query)
        if not match:
            raise ValueError("Invalid SELECT query format")

        vector_string, metadata_filter, order_by, order_dir, limit, offset = match.groups()

        vector = self._parse_vector(vector_string) if vector_string else None
        limit = int(limit) if limit else 10
        offset = int(offset) if offset else 0

        # Step 1: Perform similarity search if vector provided
        if vector is not None:
            results = self.vector_db.search(vector, k=limit + offset)
        else:
            # If no vector, get all stored vectors and metadata
            results = [
                (idx, 0.0, self.vector_db.metadata_store[idx])
                for idx in range(self.vector_db.get_vector_count())
            ]

        # Step 2: Filter by metadata (if provided)
        if metadata_filter:
            results = [
                r for r in results if metadata_filter.lower() in r[2].lower()
            ]

        # Step 3: Apply ORDER BY on metadata
        if order_by:
            if order_by.lower() == 'metadata':
                results = sorted(
                    results,
                    key=lambda x: x[2],
                    reverse=(order_dir.upper() == "DESC")
                )
            else:
                raise ValueError(f"Invalid ORDER BY field: {order_by}")

        # Step 4: Apply LIMIT and OFFSET
        results = results[offset: offset + limit]

        return results

    def _handle_insert(self, query):
        pattern = re.compile(
            r"INSERT INTO VECTORS \(VECTOR, METADATA\) VALUES \(\[(.*?)\], '(.*?)'\)",
            re.IGNORECASE
        )

        match = pattern.match(query)
        if not match:
            raise ValueError("Invalid INSERT query format")

        vector_string, metadata = match.groups()
        vector = self._parse_vector(vector_string)

        self.vector_db.add_vectors([vector], [metadata])

        return "Vector inserted successfully."

    def _handle_update(self, query):
        pattern = re.compile(
            r"UPDATE VECTORS SET VECTOR = \[(.*?)\], METADATA = '(.*?)' WHERE ID = (\d+)",
            re.IGNORECASE
        )

        match = pattern.match(query)
        if not match:
            raise ValueError("Invalid UPDATE query format")

        vector_string, metadata, vector_id = match.groups()
        vector = self._parse_vector(vector_string)

        self.vector_db.update_vector(int(vector_id), vector, metadata)

        return f"Vector with ID {vector_id} updated successfully."

    def _handle_delete(self, query):
        pattern = re.compile(
            r"DELETE FROM VECTORS WHERE ID = (\d+)",
            re.IGNORECASE
        )

        match = pattern.match(query)
        if not match:
            raise ValueError("Invalid DELETE query format")

        vector_id = int(match.group(1))
        self.vector_db.delete_vector(vector_id)

        return f"Vector with ID {vector_id} deleted successfully."

# Example Usage
if __name__ == "__main__":
    # Initialize Vector Database
    db = VectorSearch(dimension=128)

    # Add sample vectors
    vectors = np.random.random((10, 128)).astype('float32')
    metadata = [f"Sample {i}" for i in range(10)]
    db.add_vectors(vectors, metadata)

    # Create Query Processor
    processor = QueryProcessor(db)

    # SELECT with metadata filter, order, limit, and offset
    print(processor.process_query(
        "SELECT * FROM VECTORS WHERE METADATA LIKE 'Sample' ORDER BY metadata ASC LIMIT 5 OFFSET 2"
    ))

    # INSERT
    print(processor.process_query(
        "INSERT INTO VECTORS (VECTOR, METADATA) VALUES ([0.1, 0.2, ...], 'New Vector')"
    ))

    # UPDATE
    print(processor.process_query(
        "UPDATE VECTORS SET VECTOR = [0.3, 0.4, ...], METADATA = 'Updated Vector' WHERE ID = 2"
    ))

    # DELETE
    print(processor.process_query(
        "DELETE FROM VECTORS WHERE ID = 3"
    ))
