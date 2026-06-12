"""
CHROMADB - Vector Database đơn giản nhất
==========================================
- Chạy hoàn toàn local, không cần server, không cần API key
- Cài: pip install chromadb sentence-transformers

Tài liệu: https://docs.trychroma.com
"""

import chromadb
from chromadb.utils import embedding_functions

# ─────────────────────────────────────────────
# 1. Khởi tạo Client
# ─────────────────────────────────────────────
print("=" * 60)
print("1. KHỞI TẠO CLIENT")
print("=" * 60)

# Option A: In-memory (mất dữ liệu khi restart)
client_memory = chromadb.Client()
print("✓ In-memory client")

# Option B: Persistent (lưu xuống disk)
client = chromadb.PersistentClient(path="./chroma_data")
print("✓ Persistent client → ./chroma_data/")


# ─────────────────────────────────────────────
# 2. Embedding Function
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CHỌN EMBEDDING FUNCTION")
print("=" * 60)

# Option A: Dùng sentence-transformers (local, miễn phí)
# Model nhỏ ~90MB, chạy được trên CPU
ef_local = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # 384 chiều, nhanh và tốt
)
print("✓ SentenceTransformer: all-MiniLM-L6-v2")

# Option B: Dùng OpenAI embeddings (cần API key)
# ef_openai = embedding_functions.OpenAIEmbeddingFunction(
#     api_key="sk-...",
#     model_name="text-embedding-3-small"  # 1536 chiều
# )

# Option C: Để ChromaDB tự handle (dùng all-MiniLM-L6-v2 mặc định)
# Không cần truyền gì cả


# ─────────────────────────────────────────────
# 3. Tạo Collection (như "bảng" trong SQL)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. TẠO COLLECTION")
print("=" * 60)

# Xóa nếu đã tồn tại (để demo chạy lại được)
try:
    client.delete_collection("san_pham")
except Exception:
    pass

collection = client.create_collection(
    name="san_pham",
    embedding_function=ef_local,
    metadata={"hnsw:space": "cosine"}  # cosine similarity (mặc định)
    # metadata={"hnsw:space": "l2"}    # Euclidean distance
    # metadata={"hnsw:space": "ip"}    # Inner product
)
print(f"✓ Tạo collection: '{collection.name}'")


# ─────────────────────────────────────────────
# 4. Thêm dữ liệu (documents + metadata)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. THÊM DỮ LIỆU")
print("=" * 60)

# Dữ liệu mẫu: sản phẩm thương mại điện tử
documents = [
    "Laptop Dell XPS 15 màn hình OLED, chip Intel Core i9, RAM 32GB",
    "MacBook Pro M3 Pro 14 inch, chip Apple M3 Pro, RAM 18GB",
    "Điện thoại iPhone 15 Pro Max, chip A17 Pro, màn hình 6.7 inch",
    "Samsung Galaxy S24 Ultra, chip Snapdragon 8 Gen 3, camera 200MP",
    "Tai nghe Sony WH-1000XM5, chống ồn chủ động, pin 30 giờ",
    "Tai nghe AirPods Pro 2, chip H2, chống ồn thích ứng",
    "Chuột Logitech MX Master 3S, kết nối Bluetooth, 8000 DPI",
    "Bàn phím cơ Keychron K2, switch Brown, kết nối không dây",
]

metadatas = [
    {"category": "laptop",    "brand": "Dell",     "price": 45000000},
    {"category": "laptop",    "brand": "Apple",    "price": 52000000},
    {"category": "phone",     "brand": "Apple",    "price": 33000000},
    {"category": "phone",     "brand": "Samsung",  "price": 28000000},
    {"category": "headphone", "brand": "Sony",     "price": 8500000},
    {"category": "headphone", "brand": "Apple",    "price": 6500000},
    {"category": "mouse",     "brand": "Logitech", "price": 2200000},
    {"category": "keyboard",  "brand": "Keychron", "price": 2800000},
]

ids = [f"sp_{i+1:03d}" for i in range(len(documents))]

# Thêm một lúc nhiều document
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
    # embeddings=...,  # nếu bạn tự tính embedding, truyền vào đây
)
print(f"✓ Đã thêm {len(documents)} sản phẩm")
print(f"  IDs: {ids}")


# ─────────────────────────────────────────────
# 5. Tìm kiếm ngữ nghĩa (semantic search)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TÌM KIẾM NGỮ NGHĨA")
print("=" * 60)

def hien_thi_ket_qua(results, ten_query):
    print(f"\nQuery: '{ten_query}'")
    print("-" * 40)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        score = 1 - dist  # chuyển cosine distance → similarity
        print(f"  [{i+1}] Score: {score:.3f} | {meta['brand']} | {doc[:50]}...")


# Tìm sản phẩm liên quan đến "máy tính xách tay"
results = collection.query(
    query_texts=["máy tính xách tay hiệu năng cao"],
    n_results=3  # lấy 3 kết quả gần nhất
)
hien_thi_ket_qua(results, "máy tính xách tay hiệu năng cao")


# Tìm sản phẩm liên quan đến "âm thanh / nghe nhạc"
results = collection.query(
    query_texts=["tai nghe không dây chống ồn"],
    n_results=3
)
hien_thi_ket_qua(results, "tai nghe không dây chống ồn")


# ─────────────────────────────────────────────
# 6. Tìm kiếm có Filter (metadata filtering)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. TÌM KIẾM CÓ FILTER")
print("=" * 60)

# Tìm laptop của Apple
results = collection.query(
    query_texts=["laptop mạnh nhất"],
    n_results=2,
    where={"brand": "Apple"}           # filter chính xác
)
print("\nQuery: 'laptop mạnh nhất' WHERE brand='Apple'")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  → {meta['brand']} | {doc[:60]}...")


# Filter phức tạp hơn với $and, $or, $gte, $lte
results = collection.query(
    query_texts=["thiết bị công nghệ"],
    n_results=3,
    where={
        "$and": [
            {"price": {"$gte": 5000000}},    # giá >= 5 triệu
            {"price": {"$lte": 35000000}},   # giá <= 35 triệu
            {"category": {"$ne": "laptop"}}, # không phải laptop
        ]
    }
)
print("\nQuery: 'thiết bị công nghệ' WHERE 5M ≤ price ≤ 35M AND category ≠ 'laptop'")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  → {meta['category']} | {meta['price']:,}đ | {doc[:50]}...")


# Filter theo $in (thuộc danh sách)
results = collection.query(
    query_texts=["thiết bị di động"],
    n_results=5,
    where={"category": {"$in": ["phone", "headphone"]}}
)
print("\nQuery: 'thiết bị di động' WHERE category IN ['phone', 'headphone']")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  → {meta['category']} | {doc[:50]}...")


# ─────────────────────────────────────────────
# 7. Tìm kiếm trong document text (where_document)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. FILTER THEO NỘI DUNG DOCUMENT")
print("=" * 60)

results = collection.query(
    query_texts=["thiết bị Apple"],
    n_results=3,
    where_document={"$contains": "chip"}  # document phải chứa từ "chip"
)
print("\nQuery: 'thiết bị Apple' WHERE document CONTAINS 'chip'")
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    print(f"  → {doc[:60]}...")


# ─────────────────────────────────────────────
# 8. Get / Update / Delete
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GET / UPDATE / DELETE")
print("=" * 60)

# GET by ID
result = collection.get(ids=["sp_001", "sp_003"])
print("GET sp_001 và sp_003:")
for doc, meta in zip(result["documents"], result["metadatas"]):
    print(f"  → {meta['brand']}: {doc[:50]}...")

# UPDATE (upsert)
collection.update(
    ids=["sp_001"],
    metadatas=[{"category": "laptop", "brand": "Dell", "price": 43000000}]  # giảm giá
)
updated = collection.get(ids=["sp_001"])
print(f"\nSau khi update giá sp_001: {updated['metadatas'][0]['price']:,}đ")

# UPSERT (insert nếu chưa có, update nếu có)
collection.upsert(
    ids=["sp_009"],
    documents=["iPad Pro M4 11 inch, chip Apple M4, màn OLED 1000 nit"],
    metadatas=[{"category": "tablet", "brand": "Apple", "price": 22000000}]
)
print(f"\nUpsert sp_009 (iPad Pro) thành công")

# DELETE
collection.delete(ids=["sp_009"])
print("Delete sp_009 thành công")

# Kiểm tra số lượng
count = collection.count()
print(f"\nSố document hiện tại: {count}")


# ─────────────────────────────────────────────
# 9. Liệt kê tất cả Collection
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. QUẢN LÝ COLLECTION")
print("=" * 60)

# Tạo thêm một collection khác
try:
    client.delete_collection("khach_hang")
except Exception:
    pass

col2 = client.get_or_create_collection("khach_hang")

collections = client.list_collections()
print(f"Danh sách collections: {[c.name for c in collections]}")

# Lấy collection đã tồn tại
san_pham = client.get_collection("san_pham", embedding_function=ef_local)
print(f"Collection 'san_pham' có {san_pham.count()} documents")


print("\n" + "=" * 60)
print("TỔNG KẾT CHROMADB")
print("=" * 60)
print("""
✓ Ưu điểm:
  - Không cần server, cài và chạy ngay
  - API Python rất đơn giản
  - Tích hợp sẵn nhiều embedding function
  - Miễn phí hoàn toàn

✗ Nhược điểm:
  - Không scale tốt cho dữ liệu rất lớn (>1M vectors)
  - Filter không mạnh bằng Qdrant
  - Chưa có distributed mode

→ File tiếp theo: 02_pinecone_demo.py
""")
