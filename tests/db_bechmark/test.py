import os
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
from query_processer import QueryProcessor

# Optional imports - you'll need to install these
import faiss
import annoy
import weaviate
import pinecone
import singlestoredb as s2

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None

from dotenv import load_dotenv
load_dotenv()

# Constants
NUM_VECTORS = 10000  # Reduced for faster testing
VECTOR_DIM = 384  # Matching all-MiniLM-L6-v2 dimensions
TOP_K = 10


def generate_dataset():
    """Generate a test dataset with random text embeddings and metadata"""
    print("🔄 Generating dataset...")

    # Initialize sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate random text samples
    random_texts = [f"Sample text document {i}" for i in range(NUM_VECTORS)]

    # Generate embeddings
    print("⏳ Encoding text embeddings...")
    text_embeddings = model.encode(random_texts).tolist()

    # Generate random metadata
    categories = ["science", "history", "technology",
                  "sports", "health", "finance", "arts"]
    dataset = []
    for i in range(NUM_VECTORS):
        vector = text_embeddings[i]
        metadata = {
            "category": np.random.choice(categories),
            "importance": round(np.random.random(), 2),
            "original_text": random_texts[i]
        }
        dataset.append({
            "id": str(i),
            "vector": vector,
            "text": random_texts[i],
            "metadata": metadata
        })

    return dataset


def benchmark_database(db_name, setup_fn, insert_fn, query_fn, update_fn, delete_fn, dataset):
    """Benchmark a vector database for various operations"""
    print(f"\n🔍 Benchmarking {db_name}...")

    results = {
        "database": db_name,
        "total_vectors": NUM_VECTORS,
        "vector_dimension": VECTOR_DIM,
        "operations": {}
    }

    try:
        # Setup
        print("Setting up database...")
        start_time = time.time()
        db = setup_fn()
        setup_time = time.time() - start_time
        results["operations"]["setup"] = {
            "time": setup_time
        }
        print(f"Setup completed in {setup_time:.4f}s")

        # Insertion
        print("Benchmarking insertion...")
        start_time = time.time()
        insert_fn(db, dataset)
        insertion_time = time.time() - start_time
        results["operations"]["insertion"] = {
            "time": insertion_time,
            "rate": NUM_VECTORS / insertion_time if insertion_time > 0 else 0
        }
        print(
            f"Insertion completed in {insertion_time:.4f}s (Rate: {results['operations']['insertion']['rate']:.2f} vectors/s)")

        # Querying
        print("Benchmarking queries...")
        query_times = []
        for _ in range(5):
            start_time = time.time()
            query_fn(db)
            query_times.append(time.time() - start_time)

        avg_query_time = np.mean(query_times)
        results["operations"]["query"] = {
            "avg_time": avg_query_time,
            "times": query_times
        }
        print(f"Average query time: {avg_query_time:.4f}s")

        # Update
        try:
            print("Benchmarking updates...")
            start_time = time.time()
            update_fn(db, dataset)
            update_time = time.time() - start_time
            results["operations"]["update"] = {
                "time": update_time
            }
            print(f"Update completed in {update_time:.4f}s")
        except NotImplementedError:
            print("Update not implemented for this database")
            results["operations"]["update"] = {
                "time": None, "status": "Not implemented"}

        # Delete
        try:
            print("Benchmarking deletions...")
            start_time = time.time()
            delete_fn(db, dataset)
            delete_time = time.time() - start_time
            results["operations"]["delete"] = {
                "time": delete_time
            }
            print(f"Deletion completed in {delete_time:.4f}s")
        except NotImplementedError:
            print("Delete not implemented for this database")
            results["operations"]["delete"] = {
                "time": None, "status": "Not implemented"}

        return results

    except Exception as e:
        print(f"Error benchmarking {db_name}: {str(e)}")
        return {"database": db_name, "error": str(e)}

# -------------------- FAISS Implementation --------------------


def setup_faiss():
    """Set up FAISS index"""
    index = faiss.IndexFlatL2(VECTOR_DIM)
    return index


def insert_faiss(db, dataset):
    """Insert vectors into FAISS"""
    vectors = np.array([d["vector"] for d in dataset], dtype=np.float32)
    db.add(vectors)


def query_faiss(db):
    """Query FAISS index"""
    query_vector = np.random.rand(1, VECTOR_DIM).astype(np.float32)
    _, indices = db.search(query_vector, k=TOP_K)


def update_faiss(db, dataset):
    """Update vectors in FAISS (not directly supported)"""
    raise NotImplementedError("FAISS doesn't support direct updates")


def delete_faiss(db, dataset):
    """Delete vectors from FAISS"""
    raise NotImplementedError("FAISS basic version doesn't support deletion")

# -------------------- Annoy Implementation --------------------


def setup_annoy():
    """Set up Annoy index"""
    index = annoy.AnnoyIndex(VECTOR_DIM, 'angular')
    return index


def insert_annoy(db, dataset):
    """Insert vectors into Annoy"""
    for i, entry in enumerate(dataset):
        db.add_item(i, entry["vector"])
    db.build(10)  # 10 trees for better accuracy


def query_annoy(db):
    """Query Annoy index"""
    query_vector = np.random.rand(VECTOR_DIM).tolist()
    db.get_nns_by_vector(query_vector, TOP_K)


def update_annoy(db, dataset):
    """Update vectors in Annoy"""
    raise NotImplementedError("Annoy is immutable after building")


def delete_annoy(db, dataset):
    """Delete vectors from Annoy"""
    raise NotImplementedError("Annoy is immutable after building")

# -------------------- Pinecone Implementation --------------------


def setup_pinecone():
    """Set up Pinecone index"""
    if Pinecone is None:
        raise ImportError("Pinecone library not installed")

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Please set PINECONE_API_KEY environment variable")

    pc = Pinecone(api_key=api_key)

    # Index name
    index_name = "vector-benchmark"

    # Delete existing index if it exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create new index
    pc.create_index(
        name=index_name,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait for index to be ready
    time.sleep(30)  # Give more time for index initialization

    return pc.Index(index_name)


def insert_pinecone(db, dataset):
    """Insert vectors into Pinecone"""
    batch_size = 100
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        vectors = [(entry["id"], entry["vector"], entry["metadata"])
                   for entry in batch]
        db.upsert(vectors)


def query_pinecone(db):
    """Query Pinecone index"""
    query_vector = np.random.rand(VECTOR_DIM).tolist()
    db.query(vector=query_vector, top_k=TOP_K, include_metadata=True)


def update_pinecone(db, dataset):
    """Update vectors in Pinecone"""
    batch_size = 100
    update_batch = dataset[:batch_size]
    updated_vectors = [(entry["id"], np.random.rand(
        VECTOR_DIM).tolist(), entry["metadata"]) for entry in update_batch]
    db.upsert(updated_vectors)


def delete_pinecone(db, dataset):
    """Delete vectors from Pinecone"""
    ids_to_delete = [entry["id"] for entry in dataset[:100]]
    db.delete(ids=ids_to_delete)

# -------------------- SingleStore Implementation --------------------


def setup_singlestore():
    """Set up SingleStore vector database using connection string"""
    conn_str = os.getenv("SINGLESTORE_CONNECTION_STRING")

    if not conn_str:
        raise ValueError(
            "SINGLESTORE_CONNECTION_STRING environment variable is not set")

    try:
        conn = s2.connect(conn_str)
        # Verify connection by executing a simple query
        cursor = conn.cursor()
        # Drop the table if it exists
        cursor.execute("DROP TABLE IF EXISTS vectors;")
        cursor.execute("SELECT 1")
        print("✅ Successfully connected to SingleStore")
        return conn
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        raise


def insert_singlestore(db, dataset):
    """Insert vectors into SingleStore"""
    cursor = db.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS vectors (id VARCHAR(255) PRIMARY KEY, vector JSON, metadata JSON)")
    for entry in dataset:
        cursor.execute("INSERT INTO vectors (id, vector, metadata) VALUES (%s, %s, %s)",
                       (entry["id"], json.dumps(entry["vector"]), json.dumps(entry["metadata"])))
    db.commit()


def query_singlestore(db):
    """Query SingleStore vector database"""
    cursor = db.cursor()
    cursor.execute("SELECT id, vector FROM vectors LIMIT 10")
    for row in cursor.fetchall():
        print(row)
    return cursor.fetchall()


def update_singlestore(db, dataset):
    """Update vectors in SingleStore"""
    raise NotImplementedError("SingleStore doesn't support direct updates")


def delete_singlestore(db, dataset):
    """Delete vectors from SingleStore"""
    cursor = db.cursor()
    ids_to_delete = [entry["id"] for entry in dataset[:100]]
    cursor.execute("DELETE FROM vectors WHERE id IN (%s)" %
                   ','.join(['%s']*len(ids_to_delete)), tuple(ids_to_delete))
    db.commit()

# -------------------- HybridVDBMS Implementation --------------------


def benchmark_vdms():
    """Benchmark the Vector Database Management System"""
    print("\n🚀 Benchmarking HybridVDBMS...")

    # Initialize results dictionary
    results = {
        "database": "HybridVDBMS",
        "total_vectors": NUM_VECTORS,
        "vector_dimension": VECTOR_DIM,
        "operations": {}
    }

    # Generate dataset
    dataset = generate_dataset()

    # Setup
    print("Setting up database...")
    start_time = time.time()
    query_processor = QueryProcessor()
    setup_time = time.time() - start_time
    results["operations"]["setup"] = {"time": setup_time}
    print(f"Setup completed in {setup_time:.4f}s")

    # Create table
    query_processor.execute_sql(
        "CREATE TABLE vectors (vector FLOAT, metadata TEXT);")

    # Insertion Benchmark
    print("Benchmarking insertion...")
    start_time = time.time()

    # Insert vectors with metadata
    insertion_errors = 0
    for entry in dataset:
        try:
            # Convert vector to string for SQL insertion
            vector_str = ', '.join(map(str, entry['vector']))
            insert_sql = f"INSERT INTO vectors (vector, metadata) VALUES ([{vector_str}], '{entry['metadata']}');"
            result = query_processor.execute_sql(insert_sql)

            if not result.startswith('✅'):
                insertion_errors += 1
        except Exception as e:
            insertion_errors += 1

    insertion_time = time.time() - start_time
    insertion_rate = NUM_VECTORS / insertion_time if insertion_time > 0 else 0

    results["operations"]["insertion"] = {
        "time": insertion_time,
        "rate": insertion_rate,
        "errors": insertion_errors
    }
    print(
        f"Insertion completed in {insertion_time:.4f}s (Rate: {insertion_rate:.2f} vectors/s)")

    # Querying Benchmark
    print("Benchmarking queries...")
    query_times = []

    for _ in range(5):
        # Prepare a random query vector
        query_vector = np.random.rand(VECTOR_DIM).tolist()
        vector_str = ', '.join(map(str, query_vector))

        # Perform semantic search query
        start_time = time.time()
        query_sql = f"SELECT * FROM vectors WHERE VECTOR = [{vector_str}] LIMIT {TOP_K};"
        query_processor.execute_sql(query_sql)
        query_times.append(time.time() - start_time)

    avg_query_time = np.mean(query_times)
    results["operations"]["query"] = {
        "avg_time": avg_query_time,
        "times": query_times
    }
    print(f"Average query time: {avg_query_time:.4f}s")

    # Semantic Search Benchmark
    print("Benchmarking semantic search...")
    semantic_search_times = []

    for _ in range(5):
        # Random text query
        text_query = f"Sample search query {np.random.randint(1000)}"

        start_time = time.time()
        semantic_sql = f"SELECT * FROM vectors WHERE TEXT = '{text_query}' LIMIT {TOP_K};"
        query_processor.execute_sql(semantic_sql)
        semantic_search_times.append(time.time() - start_time)

    avg_semantic_time = np.mean(semantic_search_times)
    results["operations"]["semantic_search"] = {
        "avg_time": avg_semantic_time,
        "times": semantic_search_times
    }
    print(f"Average semantic search time: {avg_semantic_time:.4f}s")

    # Save results
    results_file = "vdms_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Benchmarking Complete! Results saved to {results_file}")

    return results


def create_benchmark_table(benchmark_results):
    """
    Create a tabulated summary of benchmark results

    Args:
        benchmark_results (list): List of benchmark result dictionaries

    Returns:
        str: Formatted table of benchmark results
    """
    # Prepare table data
    table_data = []
    headers = [
        "Database",
        "Setup Time (s)",
        "Insertion Time (s)",
        "Insertion Rate (vec/s)",
        "Avg Query Time (s)",
        "Semantic Search Time (s)"
    ]

    for result in benchmark_results:
        # Extract data with error handling
        try:
            ops = result.get('operations', {})
            row = [
                result.get('database', 'Unknown'),
                ops.get('setup', {}).get('time', 'N/A'),
                ops.get('insertion', {}).get('time', 'N/A'),
                ops.get('insertion', {}).get('rate', 'N/A'),
                ops.get('query', {}).get('avg_time', 'N/A'),
                ops.get('semantic_search', {}).get('avg_time', 'N/A')
            ]
            table_data.append(row)
        except Exception as e:
            print(
                f"Error processing result for {result.get('database', 'Unknown')}: {e}")

    # Generate table
    table_str = tabulate(table_data, headers=headers,
                         tablefmt="grid", floatfmt=".4f")

    return table_str


def main():
    # Generate dataset
    dataset = generate_dataset()

    # Benchmark results
    benchmark_results = []

    # Databases to benchmark (comment/uncomment as needed)
    databases = [
        # Local vector databases
        {
            "name": "FAISS",
            "setup": setup_faiss,
            "insert": insert_faiss,
            "query": query_faiss,
            "update": update_faiss,
            "delete": delete_faiss
        },
        {
            "name": "Annoy",
            "setup": setup_annoy,
            "insert": insert_annoy,
            "query": query_annoy,
            "update": update_annoy,
            "delete": delete_annoy
        }
    ]

    try:
        if Pinecone is not None:
            databases.append({
                "name": "Pinecone",
                "setup": setup_pinecone,
                "insert": insert_pinecone,
                "query": query_pinecone,
                "update": update_pinecone,
                "delete": delete_pinecone
            })
    except ImportError:
        print("Pinecone library not available, skipping Pinecone benchmark")

    # Add SingleStore-V to the benchmarking databases
    databases.append({
        "name": "SingleStore-V",
        "setup": setup_singlestore,
        "insert": insert_singlestore,
        "query": query_singlestore,
        "update": update_singlestore,
        "delete": delete_singlestore
    })

    # Benchmark HybridVDBMS first
    vdms_result = benchmark_vdms()
    benchmark_results.append(vdms_result)

    # Run benchmarks for other databases
    for db in databases:
        try:
            result = benchmark_database(
                db["name"],
                db["setup"],
                db["insert"],
                db["query"],
                db["update"],
                db["delete"],
                dataset
            )
            benchmark_results.append(result)
        except Exception as e:
            print(f"Failed to benchmark {db['name']}: {str(e)}")

    # Create benchmark summary table
    benchmark_table = create_benchmark_table(benchmark_results)

    # Print table to terminal
    print("\n📊 Vector Database Benchmark Comparison:")
    print(benchmark_table)

    # Save results to JSON
    with open("vector_db_benchmark_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Save table to text file
    with open("vector_db_benchmark_summary.txt", "w") as f:
        f.write("Vector Database Benchmark Comparison\n")
        f.write("=====================================\n\n")
        f.write(benchmark_table)

    print("\n✅ Benchmarking Complete!")
    print("Results saved to:")
    print("- vector_db_benchmark_results.json")
    print("- vector_db_benchmark_summary.txt")


if __name__ == "__main__":
    main()
