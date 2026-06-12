"""
QDRANT - Vector Database với Filter mạnh nhất
===============================================
- Có thể chạy local (in-memory hoặc Docker) hoặc dùng Qdrant Cloud
- Filter cực kỳ mạnh, hỗ trợ nested conditions
- Cài: pip install qdrant-client sentence-transformers

Chạy local với Docker:
  docker pull qdrant/qdrant
  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Tài liệu: https://qdrant.tech/documentation
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, Range, MatchValue, MatchAny,
    UpdateStatus, SearchParams
)
from sentence_transformers import SentenceTransformer
import uuid

COLLECTION_NAME = "san_pham"
DIMENSION = 384


# ─────────────────────────────────────────────
# 1. Khởi tạo Client
# ─────────────────────────────────────────────
print("=" * 60)
print("1. KHỞI TẠO CLIENT")
print("=" * 60)

# Option A: In-memory (không cần Docker, tốt để học/test)
client = QdrantClient(":memory:")
print("✓ In-memory client (không cần Docker)")

# Option B: Kết nối Docker local
# client = QdrantClient(host="localhost", port=6333)

# Option C: Qdrant Cloud
# client = QdrantClient(
#     url="https://xxx.qdrant.io:6333",
#     api_key="your-qdrant-api-key"
# )

# Option D: Persist xuống disk (không cần Docker)
# client = QdrantClient(path="./qdrant_data")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")


# ─────────────────────────────────────────────
# 2. Tạo Collection
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. TẠO COLLECTION")
print("=" * 60)

# Xóa nếu đã tồn tại
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=DIMENSION,
        distance=Distance.COSINE  # COSINE | DOT | EUCLID | MANHATTAN
    )
)
print(f"✓ Collection '{COLLECTION_NAME}' tạo thành công")


# ─────────────────────────────────────────────
# 3. Upsert Points
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. UPSERT POINTS")
print("=" * 60)

# Qdrant gọi mỗi record là "Point", có: id + vector + payload
san_pham = [
    {"text": "Laptop Dell XPS 15 OLED chip Intel Core i9 RAM 32GB SSD 1TB",
     "category": "laptop",    "brand": "Dell",     "price": 45000000, "in_stock": True,
     "tags": ["cao cấp", "làm việc", "sáng tạo"]},
    {"text": "MacBook Pro M3 Pro 14 inch chip Apple M3 Pro RAM 18GB",
     "category": "laptop",    "brand": "Apple",    "price": 52000000, "in_stock": True,
     "tags": ["cao cấp", "lập trình", "video editing"]},
    {"text": "iPhone 15 Pro Max chip A17 Pro titanium màn hình 6.7 inch",
     "category": "phone",     "brand": "Apple",    "price": 33000000, "in_stock": True,
     "tags": ["cao cấp", "nhiếp ảnh"]},
    {"text": "Samsung Galaxy S24 Ultra Snapdragon 8 Gen 3 camera 200MP S-Pen",
     "category": "phone",     "brand": "Samsung",  "price": 28000000, "in_stock": False,
     "tags": ["cao cấp", "camera", "năng suất"]},
    {"text": "Tai nghe Sony WH-1000XM5 chống ồn chủ động pin 30 giờ",
     "category": "headphone", "brand": "Sony",     "price": 8500000,  "in_stock": True,
     "tags": ["chống ồn", "làm việc", "du lịch"]},
    {"text": "AirPods Pro 2 chip H2 chống ồn thích ứng âm thanh không gian",
     "category": "headphone", "brand": "Apple",    "price": 6500000,  "in_stock": True,
     "tags": ["chống ồn", "Apple ecosystem"]},
    {"text": "Chuột Logitech MX Master 3S Bluetooth không dây 8000 DPI",
     "category": "mouse",     "brand": "Logitech", "price": 2200000,  "in_stock": True,
     "tags": ["ergonomic", "năng suất"]},
    {"text": "Bàn phím cơ Keychron K2 switch Brown TKL không dây",
     "category": "keyboard",  "brand": "Keychron", "price": 2800000,  "in_stock": True,
     "tags": ["cơ học", "năng suất", "lập trình"]},
]

# Tạo vectors
texts = [sp["text"] for sp in san_pham]
embeddings = model.encode(texts, normalize_embeddings=True)

# Tạo danh sách PointStruct
points = []
for i, (sp, emb) in enumerate(zip(san_pham, embeddings)):
    points.append(
        PointStruct(
            id=i + 1,               # ID là integer (hoặc UUID)
            vector=emb.tolist(),
            payload={               # Payload = metadata trong Qdrant
                "text": sp["text"],
                "category": sp["category"],
                "brand": sp["brand"],
                "price": sp["price"],
                "in_stock": sp["in_stock"],
                "tags": sp["tags"],
            }
        )
    )

# Upsert tất cả
operation_info = client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,  # chờ operation hoàn tất
    points=points
)
print(f"✓ Upsert {len(points)} points: {operation_info.status}")


# ─────────────────────────────────────────────
# 4. Tạo Payload Index (tăng tốc filter)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TẠO PAYLOAD INDEX")
print("=" * 60)

# Tạo index cho các field thường dùng trong filter
from qdrant_client.models import PayloadSchemaType

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="price",
    field_schema=PayloadSchemaType.INTEGER
)
print("✓ Tạo index cho 'category' và 'price'")


# ─────────────────────────────────────────────
# 5. Search cơ bản
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SEARCH CƠ BẢN")
print("=" * 60)

def tim_kiem(query_text, limit=3, query_filter=None):
    query_vec = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,        # không cần trả về vector
        score_threshold=0.0        # lọc kết quả dưới ngưỡng score
    )
    return results


results = tim_kiem("laptop hiệu năng cao")
print("\nQuery: 'laptop hiệu năng cao'")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['brand']} | {r.payload['text'][:50]}...")


# ─────────────────────────────────────────────
# 6. Filtering - Điểm mạnh nhất của Qdrant
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FILTERING - Điểm mạnh của Qdrant")
print("=" * 60)

# --- Filter theo giá trị chính xác ---
results = tim_kiem(
    "sản phẩm Apple tốt nhất",
    limit=3,
    query_filter=Filter(
        must=[
            FieldCondition(key="brand", match=MatchValue(value="Apple"))
        ]
    )
)
print("\nQuery: 'sản phẩm Apple' MUST brand='Apple'")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['category']} | {r.payload['text'][:50]}...")


# --- Filter theo khoảng số ---
results = tim_kiem(
    "thiết bị tầm trung",
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(key="price", range=Range(gte=5000000, lte=35000000))
        ]
    )
)
print("\nQuery: 'thiết bị tầm trung' MUST 5M ≤ price ≤ 35M")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['price']:,}đ | {r.payload['text'][:40]}...")


# --- Must + Must_not (AND NOT) ---
results = tim_kiem(
    "thiết bị di động cao cấp",
    limit=3,
    query_filter=Filter(
        must=[
            FieldCondition(key="in_stock", match=MatchValue(value=True))
        ],
        must_not=[
            FieldCondition(key="brand", match=MatchValue(value="Apple")),
            FieldCondition(key="category", match=MatchValue(value="laptop")),
        ]
    )
)
print("\nQuery: 'thiết bị cao cấp' MUST in_stock=True, MUST NOT brand=Apple, MUST NOT laptop")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['brand']} | {r.payload['text'][:50]}...")


# --- Match Any (IN list) ---
results = tim_kiem(
    "thiết bị âm thanh",
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchAny(any=["headphone", "speaker"])
            )
        ]
    )
)
print("\nQuery: 'thiết bị âm thanh' MUST category IN ['headphone', 'speaker']")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['category']} | {r.payload['text'][:50]}...")


# --- Filter lồng nhau (Nested) ---
results = tim_kiem(
    "laptop tốt nhất",
    limit=5,
    query_filter=Filter(
        should=[    # OR condition
            Filter(
                must=[
                    FieldCondition(key="brand", match=MatchValue(value="Apple")),
                    FieldCondition(key="price", range=Range(gte=40000000)),
                ]
            ),
            Filter(
                must=[
                    FieldCondition(key="brand", match=MatchValue(value="Dell")),
                    FieldCondition(key="price", range=Range(gte=40000000)),
                ]
            ),
        ]
    )
)
print("\nQuery: SHOULD (Apple AND price≥40M) OR (Dell AND price≥40M)")
for r in results:
    print(f"  Score: {r.score:.3f} | {r.payload['brand']} | {r.payload['price']:,}đ")


# ─────────────────────────────────────────────
# 7. Scroll - lấy tất cả records (không dùng vector)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SCROLL - LẤY RECORDS KHÔNG CẦN VECTOR")
print("=" * 60)

# Scroll giống SELECT * WHERE ... trong SQL
results, next_offset = client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=Filter(
        must=[FieldCondition(key="in_stock", match=MatchValue(value=True))]
    ),
    limit=10,
    with_payload=True,
    with_vectors=False
)
print(f"Sản phẩm còn hàng: {len(results)}")
for r in results:
    print(f"  ID:{r.id} | {r.payload['brand']} | {r.payload['category']} | {r.payload['price']:,}đ")


# ─────────────────────────────────────────────
# 8. Get, Update Payload, Delete
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GET / UPDATE / DELETE")
print("=" * 60)

# GET by IDs
results = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1, 2, 3],
    with_payload=True,
    with_vectors=False
)
print("GET id=[1,2,3]:")
for r in results:
    print(f"  {r.id}: {r.payload['brand']} - {r.payload['text'][:40]}...")

# UPDATE PAYLOAD (chỉ update metadata, giữ nguyên vector)
client.set_payload(
    collection_name=COLLECTION_NAME,
    payload={"price": 43000000, "in_stock": True},
    points=[1]  # update point id=1
)
updated = client.retrieve(collection_name=COLLECTION_NAME, ids=[1], with_payload=True)
print(f"\nSau update id=1: price={updated[0].payload['price']:,}đ")

# DELETE payload field
client.delete_payload(
    collection_name=COLLECTION_NAME,
    keys=["tags"],
    points=[1, 2]
)
print("Đã xóa field 'tags' khỏi point 1 và 2")

# DELETE points
client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=[7, 8]  # xóa point id=7 và 8
)
print("Đã xóa point 7 và 8")


# ─────────────────────────────────────────────
# 9. Named Vectors (nhiều vector type cho 1 point)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. NAMED VECTORS")
print("=" * 60)

named_vec_info = """
Qdrant hỗ trợ một point có NHIỀU vector (Named Vectors).
Ví dụ: một sản phẩm có cả vector tiêu đề và vector mô tả.

client.create_collection(
    collection_name="san_pham_v2",
    vectors_config={
        "title": VectorParams(size=384, distance=Distance.COSINE),
        "description": VectorParams(size=384, distance=Distance.COSINE),
    }
)

client.upsert(points=[
    PointStruct(
        id=1,
        vector={
            "title": title_embedding,
            "description": desc_embedding,
        },
        payload={"brand": "Apple"}
    )
])

# Tìm theo title vector
client.search(collection_name="san_pham_v2",
              query_vector=("title", query_vec), limit=3)
"""
print(named_vec_info)


# ─────────────────────────────────────────────
# 10. Collection Info
# ─────────────────────────────────────────────
print("=" * 60)
print("10. COLLECTION INFO")
print("=" * 60)

info = client.get_collection(COLLECTION_NAME)
print(f"Vectors count: {info.vectors_count}")
print(f"Status: {info.status}")
print(f"Config: dim={info.config.params.vectors.size}, metric={info.config.params.vectors.distance}")


print("\n" + "=" * 60)
print("TỔNG KẾT QDRANT")
print("=" * 60)
print("""
✓ Ưu điểm:
  - Filter mạnh nhất trong 4 loại
  - Hỗ trợ Named Vectors (multi-vector per point)
  - Rust-based → rất nhanh
  - Chạy local hoặc cloud
  - Giao diện Web UI tại localhost:6333/dashboard

✗ Nhược điểm:
  - API phức tạp hơn ChromaDB
  - Cần cài Docker để dùng full features

→ File tiếp theo: 04_weaviate_demo.py
""")
