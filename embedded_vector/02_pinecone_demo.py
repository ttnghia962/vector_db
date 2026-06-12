"""
PINECONE - Managed Cloud Vector Database
==========================================
- Fully managed, không cần tự vận hành server
- Cần tạo tài khoản: https://www.pinecone.io (có free tier)
- Cài: pip install pinecone-client sentence-transformers

Tài liệu: https://docs.pinecone.io
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np

# ─────────────────────────────────────────────
# CONFIG - thay bằng API key của bạn
# ─────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_s44DS_9srCJ42EBskRbvkNbWfh1cJq8wgZD9kJkuW91ZFbQooPydxEx3Aw9mWGGtndGuu")
INDEX_NAME = "san-pham-index"
DIMENSION = 384         # kích thước vector của all-MiniLM-L6-v2
CLOUD = "aws"
REGION = "us-east-1"    # free tier region


# ─────────────────────────────────────────────
# 1. Khởi tạo Client & Embedding Model
# ─────────────────────────────────────────────
print("=" * 60)
print("1. KHỞI TẠO")
print("=" * 60)

pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Pinecone client khởi tạo")
print("✓ Embedding model loaded")


# ─────────────────────────────────────────────
# 2. Tạo Index (tương đương collection)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. TẠO INDEX")
print("=" * 60)

# Xóa index cũ nếu tồn tại
if INDEX_NAME in [idx.name for idx in pc.list_indexes()]:
    pc.delete_index(INDEX_NAME)
    print(f"✓ Đã xóa index cũ: {INDEX_NAME}")
    time.sleep(5)  # chờ xóa hoàn tất

# Tạo Serverless Index (miễn phí, không cần cấu hình server)
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="cosine",         # cosine | euclidean | dotproduct
    spec=ServerlessSpec(
        cloud=CLOUD,
        region=REGION
    )
)

# Chờ index sẵn sàng
print(f"Đang chờ index '{INDEX_NAME}' sẵn sàng...")
while not pc.describe_index(INDEX_NAME).status.get("ready", False):
    time.sleep(2)
    print("  ...", end="", flush=True)

index = pc.Index(INDEX_NAME)
print(f"\n✓ Index '{INDEX_NAME}' sẵn sàng")


# ─────────────────────────────────────────────
# 3. Tạo Namespace (phân vùng trong Index)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. NAMESPACE (phân vùng dữ liệu)")
print("=" * 60)

# Namespace giúp phân tách dữ liệu trong cùng một index
# Ví dụ: tách theo tenant, ngôn ngữ, loại dữ liệu...
NAMESPACE_VI = "tieng-viet"
NAMESPACE_EN = "english"
print(f"Sẽ dùng namespace: '{NAMESPACE_VI}'")


# ─────────────────────────────────────────────
# 4. Tạo Embedding & Upsert dữ liệu
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. UPSERT DỮ LIỆU")
print("=" * 60)

# Dữ liệu mẫu
san_pham = [
    {"id": "sp_001", "text": "Laptop Dell XPS 15 OLED chip Intel Core i9 RAM 32GB",
     "category": "laptop", "brand": "Dell", "price": 45000000},
    {"id": "sp_002", "text": "MacBook Pro M3 Pro 14 inch chip Apple M3 Pro RAM 18GB",
     "category": "laptop", "brand": "Apple", "price": 52000000},
    {"id": "sp_003", "text": "iPhone 15 Pro Max chip A17 Pro màn hình 6.7 inch",
     "category": "phone", "brand": "Apple", "price": 33000000},
    {"id": "sp_004", "text": "Samsung Galaxy S24 Ultra Snapdragon 8 Gen 3 camera 200MP",
     "category": "phone", "brand": "Samsung", "price": 28000000},
    {"id": "sp_005", "text": "Tai nghe Sony WH-1000XM5 chống ồn chủ động pin 30 giờ",
     "category": "headphone", "brand": "Sony", "price": 8500000},
    {"id": "sp_006", "text": "AirPods Pro 2 chip H2 chống ồn thích ứng",
     "category": "headphone", "brand": "Apple", "price": 6500000},
    {"id": "sp_007", "text": "Chuột Logitech MX Master 3S Bluetooth 8000 DPI",
     "category": "mouse", "brand": "Logitech", "price": 2200000},
    {"id": "sp_008", "text": "Bàn phím cơ Keychron K2 switch Brown không dây",
     "category": "keyboard", "brand": "Keychron", "price": 2800000},
]

# Tạo embeddings cho tất cả text
texts = [sp["text"] for sp in san_pham]
embeddings = model.encode(texts, normalize_embeddings=True)  # normalize cho cosine
print(f"✓ Đã tạo {len(embeddings)} embedding vectors (dim={embeddings[0].shape[0]})")

# Chuẩn bị dữ liệu theo format Pinecone: list of (id, vector, metadata)
vectors = []
for sp, emb in zip(san_pham, embeddings):
    vectors.append({
        "id": sp["id"],
        "values": emb.tolist(),     # Pinecone cần list Python, không phải numpy
        "metadata": {
            "text": sp["text"],     # lưu text gốc vào metadata để lấy lại
            "category": sp["category"],
            "brand": sp["brand"],
            "price": sp["price"],
        }
    })

# Upsert theo batch (tối đa 100 vector / batch)
BATCH_SIZE = 100
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i + BATCH_SIZE]
    index.upsert(vectors=batch, namespace=NAMESPACE_VI)
    print(f"  Upsert batch {i//BATCH_SIZE + 1}: {len(batch)} vectors")

# Chờ index cập nhật
time.sleep(2)
stats = index.describe_index_stats()
print(f"\n✓ Index stats: {stats.namespaces}")


# ─────────────────────────────────────────────
# 5. Query - Tìm kiếm vector
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. QUERY - TÌM KIẾM")
print("=" * 60)

def tim_kiem(query_text, top_k=3, filter_dict=None, namespace=NAMESPACE_VI):
    """Hàm tìm kiếm tiện lợi"""
    # Chuyển câu hỏi thành vector
    query_vec = model.encode([query_text], normalize_embeddings=True)[0].tolist()

    result = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace=namespace,
        filter=filter_dict,     # metadata filter (optional)
        include_metadata=True,  # trả về metadata cùng kết quả
        include_values=False    # không cần trả về vector (tiết kiệm bandwidth)
    )
    return result.matches


# Query đơn giản
matches = tim_kiem("máy tính xách tay cao cấp")
print("\nQuery: 'máy tính xách tay cao cấp'")
for m in matches:
    print(f"  Score: {m.score:.3f} | {m.metadata['brand']} | {m.metadata['text'][:50]}...")


# Query với metadata filter
matches = tim_kiem(
    "thiết bị Apple tốt nhất",
    top_k=3,
    filter_dict={"brand": {"$eq": "Apple"}}
)
print("\nQuery: 'thiết bị Apple tốt nhất' WHERE brand='Apple'")
for m in matches:
    print(f"  Score: {m.score:.3f} | {m.metadata['category']} | {m.metadata['text'][:50]}...")


# ─────────────────────────────────────────────
# 6. Metadata Filtering - đầy đủ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. METADATA FILTERING")
print("=" * 60)

# Filter theo nhiều điều kiện
matches = tim_kiem(
    "thiết bị công nghệ",
    top_k=5,
    filter_dict={
        "$and": [
            {"price": {"$gte": 5000000}},
            {"price": {"$lte": 35000000}},
            {"category": {"$in": ["phone", "headphone"]}},
        ]
    }
)
print("\nQuery với filter: 5M ≤ price ≤ 35M, category IN [phone, headphone]")
for m in matches:
    print(f"  Score: {m.score:.3f} | {m.metadata['category']} | {m.metadata['price']:,}đ")


# ─────────────────────────────────────────────
# 7. Fetch, Update, Delete
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FETCH / UPDATE / DELETE")
print("=" * 60)

# FETCH by ID
result = index.fetch(ids=["sp_001", "sp_002"], namespace=NAMESPACE_VI)
print("FETCH sp_001 và sp_002:")
for vid, vec_data in result.vectors.items():
    print(f"  {vid}: {vec_data.metadata['brand']} | {vec_data.metadata['text'][:40]}...")

# UPDATE metadata (upsert lại với vector cũ)
# Pinecone không có update riêng → upsert lại cùng ID
old_vec = result.vectors["sp_001"].values
index.upsert(
    vectors=[{
        "id": "sp_001",
        "values": old_vec,
        "metadata": {
            "text": "Laptop Dell XPS 15 OLED chip Intel Core i9 RAM 32GB",
            "category": "laptop",
            "brand": "Dell",
            "price": 43000000  # giảm giá
        }
    }],
    namespace=NAMESPACE_VI
)
print("\nĐã update giá sp_001 xuống 43,000,000đ")

# DELETE by ID
index.delete(ids=["sp_007", "sp_008"], namespace=NAMESPACE_VI)
print("Đã delete sp_007 và sp_008")

# DELETE toàn bộ namespace
# index.delete(delete_all=True, namespace=NAMESPACE_VI)


# ─────────────────────────────────────────────
# 8. Sparse-Dense / Hybrid Search (Pinecone khác biệt)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. HYBRID SEARCH (Dense + Sparse)")
print("=" * 60)

hybrid_info = """
Pinecone hỗ trợ Hybrid Search = Dense vector + Sparse vector (BM25)
→ Kết hợp semantic search + keyword matching

Cần tạo index với metric="dotproduct" và có sparse values:

index.upsert(vectors=[{
    "id": "doc_1",
    "values": dense_embedding,          # semantic
    "sparse_values": {                  # keyword
        "indices": [102, 305, 811],
        "values": [0.5, 1.2, 0.3]
    },
    "metadata": {...}
}])

index.query(
    vector=dense_query_vec,
    sparse_vector={                     # BM25 query
        "indices": [102, 305],
        "values": [0.8, 0.4]
    },
    top_k=5
)
"""
print(hybrid_info)


# ─────────────────────────────────────────────
# 9. Dọn dẹp
# ─────────────────────────────────────────────
print("=" * 60)
print("9. DỌN DẸP")
print("=" * 60)

# pc.delete_index(INDEX_NAME)  # uncomment để xóa index
print(f"Index '{INDEX_NAME}' vẫn còn (comment dòng delete để giữ)")


print("\n" + "=" * 60)
print("TỔNG KẾT PINECONE")
print("=" * 60)
print("""
✓ Ưu điểm:
  - Fully managed, không lo ops/scaling
  - Hỗ trợ Hybrid Search (dense + sparse)
  - Namespace để multi-tenancy
  - Stable API, production-ready

✗ Nhược điểm:
  - Có chi phí khi scale (free tier giới hạn)
  - Không có self-hosted option
  - Filter không mạnh bằng Qdrant

→ File tiếp theo: 03_qdrant_demo.py
""")
