"""
VECTOR DATABASE LEARNING PATH
==============================
Thứ tự học:

  00_khai_niem_vector_db.py  → Khái niệm cơ bản, cosine similarity, so sánh 4 DB
  01_chromadb_demo.py        → ChromaDB: đơn giản nhất, chạy local ngay
  02_pinecone_demo.py        → Pinecone: managed cloud, cần API key
  03_qdrant_demo.py          → Qdrant: filter mạnh nhất, chạy local hoặc cloud
  04_weaviate_demo.py        → Weaviate: schema-based, hybrid search, GraphQL
  05_rag_tong_hop.py         → RAG pipeline thực tế: VectorDB + LLM

Cài dependencies:
  pip install -r requirements.txt

Bắt đầu nhanh (không cần API key, không cần Docker):
  python 00_khai_niem_vector_db.py
  python 01_chromadb_demo.py
  python 03_qdrant_demo.py      # in-memory mode
  python 04_weaviate_demo.py    # embedded mode
"""
