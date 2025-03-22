# 📂 Vector SQL Processor

A CLI-based SQL-like interface on top of a vector database that enables regular SQL users to work with vector data seamlessly.

Built with:
- `hnswlib` for high-performance vector similarity search
- `sentence-transformers` for semantic text embeddings
- `tabulate` for pretty table outputs

---

## ✨ Features

### ✅ SQL-like Query Interface
- `CREATE TABLE <name> (col1, col2, ...)` – Schema parsing
- `INSERT INTO ... (vector, metadata)` – Add raw vector data
- `INSERT INTO ... (text, metadata)` – Add text that is embedded to a vector
- `SELECT * FROM ... WHERE VECTOR = [...] LIMIT N` – Vector similarity search
- `SELECT * FROM ... WHERE TEXT = 'query' LIMIT N` – Semantic search via text
- `SELECT * FROM VECTORS;` – List all vectors and their metadata

### 🧵 Semantic Search
Use text instead of vectors:
```sql
SELECT * FROM VECTORS WHERE TEXT = 'apple pie recipe' LIMIT 3;
```
Internally, the system converts the text into a vector using a transformer model.

### 📃 Persistent Storage
- Automatically saves index and metadata to disk.
- Automatically loads on startup.

### ✨ Auto Dimension Inference
- No need to specify vector dimension manually.
- Automatically inferred on first `INSERT` or `TEXT` embedding.

### 📊 Tabular Results
All results are printed as formatted tables using `tabulate`.

---

## 📚 Usage Guide

### ✅ 1. Install Dependencies
```bash
pip install hnswlib sentence-transformers tabulate numpy
```

### ✅ 2. Run the CLI
```bash
python query_processor.py
```

### ✅ 3. Example Queries

#### Create Table
```sql
CREATE TABLE VECTORS (id INT, vector TEXT, metadata TEXT);
```

#### Insert Raw Vector
```sql
INSERT INTO VECTORS (vector, metadata) VALUES ([0.1, 0.2, 0.3], 'apple');
```

#### Insert Text (Semantic Mode)
```sql
INSERT INTO VECTORS (text, metadata) VALUES ('apple pie recipe', 'apple pie recipe');
```

#### Search with Vector
```sql
SELECT * FROM VECTORS WHERE VECTOR = [0.1, 0.2, 0.3] LIMIT 2;
```

#### Search with Text (Semantic Search)
```sql
SELECT * FROM VECTORS WHERE TEXT = 'how to bake a pie' LIMIT 3;
```

#### List All Vectors
```sql
SELECT * FROM VECTORS;
```

#### Exit the CLI
```sql
exit
```

---

## 🔧 Internal Storage
- `./store/vectors.hnsw` – HNSW index file
- `./store/vectors.hnsw.meta` – Metadata file (vector IDs → metadata)
- `./store/vectors.hnsw.config` – Stores the inferred dimension for loading

---

## ⚡ Tips
- Run `rm -rf store/` to reset your database.
- Avoid mixing vector dimensions (use only raw vectors **or** semantic inserts in one DB).
- Use semantic mode (`TEXT`) for most natural usage!

---

## 🎓 Contributors
- Gurjot Singh
- Saqib Shah
- Swara Desai
- Swagi Desai
- Jaideep Singh
- Dakshitkumar Kamaria

---

## 🌟 Future Work
- `DELETE` and `UPDATE` support
- Hybrid queries (e.g., `TEXT = ... AND category = ...`)
- REST API wrapper
- Multiple index support for different dimensions

---

## 📊 Project Structure
```
vdbms/
├── query_processor.py         # Main CLI interface
├── VectorSearch.py            # Vector DB wrapper (HNSWlib)
└── store/                     # Auto-generated index, metadata
```

---

## 🚀 License
MIT (or project-specific if academic)

