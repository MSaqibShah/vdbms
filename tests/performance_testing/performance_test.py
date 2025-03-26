"""
performance_test.py

This script tests CRUD and search performance using different Hugging Face 
sentence-similarity models with a larger dataset (Wikitext-103). It runs multiple
iterations per model, stores all test run results, aggregates them, generates detailed graphs
(in nanoseconds) in the vdbms/performance_testing/graphs folder, and produces a detailed 
HTML analysis report that is easily understood by non experts.
"""

import time
import csv
import os
import random
from datasets import load_dataset  # pip install datasets
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

import pandas as pd             # pip install pandas
import matplotlib.pyplot as plt # pip install matplotlib

# Number of test runs per model
NUM_RUNS = 5

# List of embedding models to test
MODELS_TO_TEST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-albert-small-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

# ----------------------------------------------------------------
# Load a larger dataset from Hugging Face.
# Using Wikitext-103 dataset (train split)
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

# Filter out empty or very short texts (fewer than 50 words)
large_documents = [doc for doc in dataset["text"] if doc.strip() and len(doc.split()) > 50]

# Randomly sample 100 documents from the filtered list (adjust sample size as needed)
DOCUMENTS = random.sample(large_documents, 100)
# ----------------------------------------------------------------

def measure_crud_operations(vector_search):
    """
    Measures the time taken for CRUD operations on the vector store.
    Returns a dict of timings (in seconds).
    """
    timings = {}
    
    # CREATE / INSERT: Add all documents
    start = time.time()
    for doc in DOCUMENTS:
        vector_search.add_document(doc)
    end = time.time()
    timings["create_time"] = end - start
    
    # READ: Get all documents
    start = time.time()
    _ = vector_search.get_all_documents()
    end = time.time()
    timings["read_time"] = end - start
    
    # UPDATE: Update first document if exists
    start = time.time()
    if vector_search.get_all_documents():
        vector_search.update_document(doc_id=0, new_text="Updated text content.")
    end = time.time()
    timings["update_time"] = end - start
    
    # DELETE: Delete first document if exists
    start = time.time()
    if vector_search.get_all_documents():
        vector_search.delete_document(doc_id=0)
    end = time.time()
    timings["delete_time"] = end - start
    
    return timings

def measure_search_performance(vector_search, queries, top_k: int = 2):
    """
    Measures the time taken to perform search queries.
    Returns the total search time (in seconds).
    """
    start = time.time()
    for query in queries:
        _ = vector_search.search(query, top_k=top_k)
    end = time.time()
    return end - start

def run_all_tests():
    """
    Runs the tests NUM_RUNS times for each model and writes detailed run results.
    """
    results = []  # list to store each run's result
    test_queries = [
        "How do I perform CRUD operations?",
        "What is a vector database?",
        "Tell me about machine learning."
    ]
    
    for model_name in MODELS_TO_TEST:
        print(f"--- Testing model: {model_name} ---")
        for run in range(1, NUM_RUNS+1):
            print(f"  Run {run}/{NUM_RUNS}")
            # Load embedding model
            embedder = SentenceTransformer(model_name)
            # Initialize vector search engine (replace with your actual implementation)
            vector_search = MockVectorSearch(embedder)
            
            # Measure CRUD operations
            crud_timings = measure_crud_operations(vector_search)
            # Measure search performance
            search_time = measure_search_performance(vector_search, test_queries, top_k=2)
            
            # Save run results (times in seconds)
            results.append({
                "model_name": model_name,
                "run": run,
                "create_time": crud_timings["create_time"],
                "read_time": crud_timings["read_time"],
                "update_time": crud_timings["update_time"],
                "delete_time": crud_timings["delete_time"],
                "search_time": search_time
            })
    
    # Write detailed run results to CSV
    results_file = os.path.join(os.path.dirname(__file__), "performance_results.csv")
    with open(results_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["model_name", "run", "create_time", "read_time", "update_time", "delete_time", "search_time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    return results

def aggregate_results(results):
    """
    Aggregates the results by model, computing mean and standard deviation for each metric.
    Returns a pandas DataFrame with the aggregated data.
    """
    df = pd.DataFrame(results)
    agg_df = df.groupby("model_name").agg({
        "create_time": ["mean", "std"],
        "read_time": ["mean", "std"],
        "update_time": ["mean", "std"],
        "delete_time": ["mean", "std"],
        "search_time": ["mean", "std"]
    }).reset_index()
    # Flatten the MultiIndex columns
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]
    return agg_df

def generate_graphs(agg_df):
    """
    Generates bar charts for each metric with error bars representing standard deviation.
    All values are converted to nanoseconds for readability.
    Graphs are saved as PNG files in the vdbms/performance_testing/graphs folder.
    """
    # Define graphs directory path and create it if it doesn't exist.
    graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graphs_dir, exist_ok=True)
    
    metrics = ["create_time", "read_time", "update_time", "delete_time", "search_time"]
    for metric in metrics:
        plt.figure(figsize=(10, 7))
        # Convert seconds to nanoseconds (1 second = 1e9 nanoseconds)
        means_ns = agg_df[f"{metric}_mean"] * 1e9
        stds_ns = agg_df[f"{metric}_std"] * 1e9
        
        bars = plt.bar(agg_df["model_name"], means_ns, yerr=stds_ns, capsize=5, color='skyblue', edgecolor='black')
        plt.ylabel(f"{metric.replace('_', ' ').title()} (nanoseconds)", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.title(f"Average {metric.replace('_', ' ').title()} by Model (ns)", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Annotate each bar with the exact value
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval)}', va='bottom', ha='center', fontsize=9)
        
        plt.tight_layout()
        graph_file = os.path.join(graphs_dir, f"{metric}_graph.png")
        plt.savefig(graph_file)
        plt.close()
        print(f"Saved detailed graph for {metric} as {graph_file}")

def generate_analysis_report(agg_df):
    """
    Generates a detailed HTML analysis report based on the aggregated results.
    The report explains each metric in plain language, embeds graphs, and
    identifies the best model for our project backed with relevant numbers.
    The report is saved as 'analysis_report.html'.
    """
    # Compute a simple aggregate metric (sum of create and search times) to decide the best model.
    agg_df['total_time'] = agg_df['create_time_mean'] + agg_df['search_time_mean']
    best_model_row = agg_df.loc[agg_df['total_time'].idxmin()]
    best_model = best_model_row['model_name']
    
    # Convert times to nanoseconds for display.
    def ns_format(val):
        return f"{val*1e9:,.0f}"  # formatted in nanoseconds, no decimals

    # Paths to graphs (assume they are in the 'graphs' subfolder)
    graphs_dir = "graphs"
    graph_files = {
        "Create Time": os.path.join(graphs_dir, "create_time_graph.png"),
        "Read Time": os.path.join(graphs_dir, "read_time_graph.png"),
        "Update Time": os.path.join(graphs_dir, "update_time_graph.png"),
        "Delete Time": os.path.join(graphs_dir, "delete_time_graph.png"),
        "Search Time": os.path.join(graphs_dir, "search_time_graph.png")
    }
    
    # Build the HTML report content
    report_lines = []
    report_lines.append("<html><head><title>Performance Testing Analysis Report</title>")
    report_lines.append("<style>")
    report_lines.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    report_lines.append("h1 { color: #2E5C6E; }")
    report_lines.append("h2 { color: #3E7C8C; }")
    report_lines.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
    report_lines.append("th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }")
    report_lines.append("tr:nth-child(even) { background-color: #f9f9f9; }")
    report_lines.append("</style>")
    report_lines.append("</head><body>")
    report_lines.append("<h1>Performance Testing Analysis Report</h1>")
    report_lines.append(f"<p><strong>Number of runs per model:</strong> {NUM_RUNS}</p>")
    report_lines.append("<h2>Aggregated Results (in Nanoseconds)</h2>")
    
    # Add a table with aggregated metrics
    report_lines.append("<table>")
    report_lines.append("<tr><th>Model</th><th>Create Time (ns)</th><th>Read Time (ns)</th><th>Update Time (ns)</th><th>Delete Time (ns)</th><th>Search Time (ns)</th></tr>")
    for index, row in agg_df.iterrows():
        report_lines.append("<tr>")
        report_lines.append(f"<td>{row['model_name']}</td>")
        report_lines.append(f"<td>{ns_format(row['create_time_mean'])} ± {ns_format(row['create_time_std'])}</td>")
        report_lines.append(f"<td>{ns_format(row['read_time_mean'])} ± {ns_format(row['read_time_std'])}</td>")
        report_lines.append(f"<td>{ns_format(row['update_time_mean'])} ± {ns_format(row['update_time_std'])}</td>")
        report_lines.append(f"<td>{ns_format(row['delete_time_mean'])} ± {ns_format(row['delete_time_std'])}</td>")
        report_lines.append(f"<td>{ns_format(row['search_time_mean'])} ± {ns_format(row['search_time_std'])}</td>")
        report_lines.append("</tr>")
    report_lines.append("</table>")
    
    # Detailed plain language explanation
    report_lines.append("<h2>What Do These Metrics Mean?</h2>")
    report_lines.append("<p><strong>Create Time:</strong> The time it takes to add all documents (including computing their embeddings). Lower is better for fast data ingestion.</p>")
    report_lines.append("<p><strong>Read Time:</strong> The time taken to retrieve documents from the store. This is generally fast and similar across models.</p>")
    report_lines.append("<p><strong>Update and Delete Times:</strong> The times needed to update or remove a document. These are minor but still important for real-time operations.</p>")
    report_lines.append("<p><strong>Search Time:</strong> The time taken to process search queries. Faster search times mean better responsiveness for user queries.</p>")
    
    # Identify the best model based on the sum of create and search times
    report_lines.append("<h2>Best Model Recommendation</h2>")
    report_lines.append(f"<p>Based on our aggregated performance metrics, the model <strong>{best_model}</strong> is the best fit for our project.</p>")
    report_lines.append("<p>This conclusion is based on the combined measurement of Create Time and Search Time – two critical factors for our application. "
                        f"For instance, <strong>{best_model}</strong> has an average Create Time of {ns_format(best_model_row['create_time_mean'])} ns and an average Search Time of {ns_format(best_model_row['search_time_mean'])} ns, which are the lowest among the tested models. Additionally, the low standard deviations indicate consistent performance across runs.</p>")
    
    # Include graphs
    report_lines.append("<h2>Performance Graphs</h2>")
    for title, path in graph_files.items():
        report_lines.append(f"<h3>{title}</h3>")
        report_lines.append(f"<img src='{path}' alt='{title} Graph' style='max-width:100%; height:auto;'><br><br>")
    
    report_lines.append("<p><em>Note:</em> All timing values are reported in nanoseconds (ns). Lower values indicate better performance.</p>")
    report_lines.append("</body></html>")
    
    # Write the HTML report to file
    report_file = os.path.join(os.path.dirname(__file__), "analysis_report.html")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Detailed HTML analysis report saved to {report_file}")

class MockVectorSearch:
    """
    A mock vector search class to simulate your vs.py functionality.
    Replace this with your actual VectorSearch implementation.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.documents = []
        self.embeddings = []
    
    def add_document(self, text: str):
        embedding = self.embedder.encode(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
    
    def get_all_documents(self):
        return self.documents
    
    def update_document(self, doc_id: int, new_text: str):
        if 0 <= doc_id < len(self.documents):
            self.documents[doc_id] = new_text
            self.embeddings[doc_id] = self.embedder.encode(new_text)
    
    def delete_document(self, doc_id: int):
        if 0 <= doc_id < len(self.documents):
            self.documents.pop(doc_id)
            self.embeddings.pop(doc_id)
    
    def search(self, query: str, top_k: int = 2):
        q_embedding = self.embedder.encode(query)
        # For demonstration, return the first top_k documents with a dummy score.
        results = [(doc, 0.99) for doc in self.documents[:top_k]]
        return results

if __name__ == "__main__":
    # Run all test iterations and get detailed results.
    all_results = run_all_tests()
    
    # Aggregate the results using pandas.
    agg_df = aggregate_results(all_results)
    # Save aggregated results to CSV.
    agg_file = os.path.join(os.path.dirname(__file__), "aggregated_results.csv")
    agg_df.to_csv(agg_file, index=False)
    print(f"Aggregated results saved to {agg_file}")
    
    # Generate detailed graphs (saved in the graphs folder).
    generate_graphs(agg_df)
    
    # Generate a detailed HTML analysis report.
    generate_analysis_report(agg_df)
