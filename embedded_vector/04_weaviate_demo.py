"""
WEAVIATE - Vector Database với GraphQL & Schema
================================================
- Schema-based: định nghĩa rõ kiểu dữ liệu như OOP
- GraphQL API rất mạnh
- Tích hợp nhiều vectorizer (OpenAI, Cohere, HuggingFace...)
- Cài: pip install weaviate-client sentence-transformers

Chạy local với Docker:
  docker pull semitechnologies/weaviate
  docker run -p 8080:8080 -p 50051:50051 semitechnologies/weaviate

Tài liệu: https://weaviate.io/developers/weaviate
"""

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from sentence_transformers import SentenceTransformer
import numpy as np

COLLECTION_NAME = "SanPham"   # Weaviate dùng PascalCase cho collection


# ─────────────────────────────────────────────
# 1. Khởi tạo Client
# ─────────────────────────────────────────────
print("=" * 60)
print("1. KHỞI TẠO CLIENT")
print("=" * 60)

# Option A: Weaviate Embedded (chạy local, không cần Docker)
# Thích hợp để học/test
client = weaviate.connect_to_embedded()
print("✓ Weaviate Embedded (chạy local, không cần Docker)")

# Option B: Kết nối Docker local
# client = weaviate.connect_to_local(
#     host="localhost",
#     port=8080,
#     grpc_port=50051
# )

# Option C: Weaviate Cloud Services (WCS)
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url="https://xxx.weaviate.network",
#     auth_credentials=weaviate.auth.AuthApiKey("your-api-key")
# )

model = SentenceTransformer("all-MiniLM-L6-v2")
print("✓ Embedding model loaded")


# ─────────────────────────────────────────────
# 2. Tạo Collection với Schema
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. TẠO COLLECTION VỚI SCHEMA")
print("=" * 60)

# Xóa collection cũ nếu tồn tại
if client.collections.exists(COLLECTION_NAME):
    client.collections.delete(COLLECTION_NAME)

# Tạo collection với schema rõ ràng (điểm khác biệt của Weaviate)
collection = client.collections.create(
    name=COLLECTION_NAME,

    # Cấu hình vectorizer - "none" vì ta tự cung cấp vector
    vectorizer_config=Configure.Vectorizer.none(),
    # Nếu muốn Weaviate tự vectorize, dùng:
    # vectorizer_config=Configure.Vectorizer.text2vec_openai()
    # vectorizer_config=Configure.Vectorizer.text2vec_huggingface(model="...")

    # Metric tính similarity
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE,
        ef=128,              # accuracy/speed tradeoff (higher = more accurate)
        max_connections=64,  # HNSW parameter
    ),

    # Định nghĩa Properties (fields) - strongly typed
    properties=[
        Property(name="text",      data_type=DataType.TEXT),       # full-text
        Property(name="category",  data_type=DataType.TEXT,
                 index_filterable=True, index_searchable=True),
        Property(name="brand",     data_type=DataType.TEXT,
                 index_filterable=True),
        Property(name="price",     data_type=DataType.INT,
                 index_filterable=True, index_range_filters=True),
        Property(name="in_stock",  data_type=DataType.BOOL,
                 index_filterable=True),
        Property(name="tags",      data_type=DataType.TEXT_ARRAY),
        Property(name="rating",    data_type=DataType.NUMBER),
    ]
)
print(f"✓ Collection '{COLLECTION_NAME}' tạo thành công với schema")


# ─────────────────────────────────────────────
# 3. Insert Objects
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. INSERT OBJECTS")
print("=" * 60)

san_pham_data = [
    {"text": "Laptop Dell XPS 15 OLED chip Intel Core i9 RAM 32GB SSD 1TB",
     "category": "laptop",    "brand": "Dell",     "price": 45000000,
     "in_stock": True,  "rating": 4.5, "tags": ["cao cấp", "làm việc"]},
    {"text": "MacBook Pro M3 Pro 14 inch chip Apple M3 Pro RAM 18GB",
     "category": "laptop",    "brand": "Apple",    "price": 52000000,
     "in_stock": True,  "rating": 4.8, "tags": ["cao cấp", "lập trình"]},
    {"text": "iPhone 15 Pro Max chip A17 Pro titanium màn hình 6.7 inch",
     "category": "phone",     "brand": "Apple",    "price": 33000000,
     "in_stock": True,  "rating": 4.7, "tags": ["cao cấp", "nhiếp ảnh"]},
    {"text": "Samsung Galaxy S24 Ultra Snapdragon 8 Gen 3 camera 200MP",
     "category": "phone",     "brand": "Samsung",  "price": 28000000,
     "in_stock": False, "rating": 4.6, "tags": ["camera", "năng suất"]},
    {"text": "Tai nghe Sony WH-1000XM5 chống ồn chủ động pin 30 giờ",
     "category": "headphone", "brand": "Sony",     "price": 8500000,
     "in_stock": True,  "rating": 4.7, "tags": ["chống ồn", "du lịch"]},
    {"text": "AirPods Pro 2 chip H2 chống ồn thích ứng âm thanh không gian",
     "category": "headphone", "brand": "Apple",    "price": 6500000,
     "in_stock": True,  "rating": 4.5, "tags": ["chống ồn", "wireless"]},
    {"text": "Chuột Logitech MX Master 3S Bluetooth không dây 8000 DPI",
     "category": "mouse",     "brand": "Logitech", "price": 2200000,
     "in_stock": True,  "rating": 4.4, "tags": ["ergonomic", "năng suất"]},
    {"text": "Bàn phím cơ Keychron K2 switch Brown TKL không dây",
     "category": "keyboard",  "brand": "Keychron", "price": 2800000,
     "in_stock": True,  "rating": 4.3, "tags": ["cơ học", "lập trình"]},
]

texts = [sp["text"] for sp in san_pham_data]
embeddings = model.encode(texts, normalize_embeddings=True)

# Batch insert (dùng context manager để tối ưu)
with collection.batch.dynamic() as batch:
    for sp, emb in zip(san_pham_data, embeddings):
        batch.add_object(
            properties={k: v for k, v in sp.items()},
            vector=emb.tolist()   # cung cấp vector thủ công
        )

# Kiểm tra kết quả
count = collection.aggregate.over_all(total_count=True).total_count
print(f"✓ Đã insert {count} objects")


# ─────────────────────────────────────────────
# 4. Near Vector Search (Semantic Search)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. NEAR VECTOR SEARCH")
print("=" * 60)

def lay_vec(text):
    return model.encode([text], normalize_embeddings=True)[0].tolist()

# Tìm kiếm bằng vector
results = collection.query.near_vector(
    near_vector=lay_vec("laptop hiệu năng cao"),
    limit=3,
    return_metadata=MetadataQuery(distance=True, score=True),
    return_properties=["text", "brand", "category", "price"]
)

print("\nnear_vector: 'laptop hiệu năng cao'")
for obj in results.objects:
    dist = obj.metadata.distance
    print(f"  Dist: {dist:.3f} | {obj.properties['brand']} | {obj.properties['text'][:50]}...")


# ─────────────────────────────────────────────
# 5. Near Text Search (Weaviate tự vectorize)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. NEAR TEXT SEARCH (khi dùng built-in vectorizer)")
print("=" * 60)

# near_text hoạt động khi collection có vectorizer (vd: text2vec_openai)
# Vì ta dùng vectorizer.none(), ta cần near_vector
# Đây là ví dụ minh họa cách dùng near_text:
near_text_example = """
# Khi tạo collection với vectorizer:
#   vectorizer_config=Configure.Vectorizer.text2vec_openai()

# Thì có thể dùng near_text (Weaviate tự vectorize):
results = collection.query.near_text(
    query="laptop mạnh nhất cho lập trình",
    limit=3,
    return_metadata=MetadataQuery(distance=True)
)
"""
print(near_text_example)
print("→ Vì đang dùng vectorizer.none(), ta dùng near_vector thay thế")


# ─────────────────────────────────────────────
# 6. Filtering - Weaviate Filter API
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FILTERING")
print("=" * 60)

# Filter chính xác
results = collection.query.near_vector(
    near_vector=lay_vec("sản phẩm Apple"),
    limit=3,
    filters=Filter.by_property("brand").equal("Apple"),
    return_properties=["text", "brand", "category", "price"]
)
print("\nnear_vector WHERE brand='Apple'")
for obj in results.objects:
    print(f"  {obj.properties['category']} | {obj.properties['text'][:50]}...")


# Filter phạm vi số
results = collection.query.near_vector(
    near_vector=lay_vec("thiết bị tầm trung"),
    limit=5,
    filters=(
        Filter.by_property("price").greater_or_equal(5000000) &
        Filter.by_property("price").less_or_equal(35000000) &
        Filter.by_property("in_stock").equal(True)
    ),
    return_properties=["text", "brand", "price"]
)
print("\nnear_vector WHERE 5M ≤ price ≤ 35M AND in_stock=True")
for obj in results.objects:
    print(f"  {obj.properties['price']:,}đ | {obj.properties['text'][:50]}...")


# Filter OR
results = collection.query.near_vector(
    near_vector=lay_vec("thiết bị nghe nhạc"),
    limit=5,
    filters=(
        Filter.by_property("category").equal("headphone") |
        Filter.by_property("category").equal("speaker")
    ),
    return_properties=["text", "brand", "category"]
)
print("\nnear_vector WHERE category='headphone' OR category='speaker'")
for obj in results.objects:
    print(f"  {obj.properties['category']} | {obj.properties['text'][:50]}...")


# ─────────────────────────────────────────────
# 7. BM25 Search (keyword/full-text search)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. BM25 FULL-TEXT SEARCH (keyword)")
print("=" * 60)

# Tìm kiếm theo từ khóa (không dùng vector)
results = collection.query.bm25(
    query="chip Apple M3",
    limit=3,
    return_metadata=MetadataQuery(score=True),
    return_properties=["text", "brand", "category"]
)
print("\nBM25: 'chip Apple M3'")
for obj in results.objects:
    print(f"  BM25 score: {obj.metadata.score:.3f} | {obj.properties['text'][:50]}...")


# ─────────────────────────────────────────────
# 8. Hybrid Search (vector + BM25)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. HYBRID SEARCH (vector + BM25)")
print("=" * 60)

# Kết hợp semantic + keyword search
results = collection.query.hybrid(
    query="tai nghe chống ồn Sony",
    vector=lay_vec("tai nghe chống ồn Sony"),
    alpha=0.75,      # 0=BM25 only, 1=vector only, 0.75=mostly vector
    limit=3,
    return_metadata=MetadataQuery(score=True),
    return_properties=["text", "brand", "category", "price"]
)
print("\nHybrid (alpha=0.75): 'tai nghe chống ồn Sony'")
for obj in results.objects:
    print(f"  Score: {obj.metadata.score:.3f} | {obj.properties['brand']} | {obj.properties['text'][:50]}...")


# ─────────────────────────────────────────────
# 9. Aggregate (thống kê)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. AGGREGATE - THỐNG KÊ")
print("=" * 60)

from weaviate.classes.aggregate import GroupByAggregate

# Đếm theo category
agg = collection.aggregate.over_all(
    group_by=GroupByAggregate(prop="category"),
    total_count=True
)
print("Số sản phẩm theo category:")
for group in agg.groups:
    print(f"  {group.grouped_by.value}: {group.total_count}")

# Thống kê price
from weaviate.classes.aggregate import Metrics
agg2 = collection.aggregate.over_all(
    return_metrics=[
        Metrics("price").number(
            sum_=True, mean=True, minimum=True, maximum=True
        )
    ]
)
price_metrics = agg2.properties["price"]
print(f"\nThống kê giá:")
print(f"  Min: {price_metrics.minimum:,.0f}đ")
print(f"  Max: {price_metrics.maximum:,.0f}đ")
print(f"  Mean: {price_metrics.mean:,.0f}đ")


# ─────────────────────────────────────────────
# 10. CRUD - Get, Update, Delete
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("10. CRUD")
print("=" * 60)

# GET tất cả (dùng fetch_objects)
results = collection.query.fetch_objects(
    limit=3,
    return_properties=["text", "brand", "price"]
)
print("Fetch 3 objects đầu tiên:")
for obj in results.objects:
    print(f"  UUID: {str(obj.uuid)[:8]}... | {obj.properties['brand']}")

# Lưu UUID để update/delete
first_uuid = results.objects[0].uuid

# UPDATE object
collection.data.update(
    uuid=first_uuid,
    properties={"price": 43000000}  # chỉ update field này
)
updated = collection.query.fetch_object_by_id(first_uuid)
print(f"\nSau update: price={updated.properties['price']:,}đ")

# DELETE object
last_uuid = results.objects[-1].uuid
collection.data.delete_by_id(last_uuid)
print(f"Đã delete object {str(last_uuid)[:8]}...")

# DELETE theo filter
collection.data.delete_many(
    where=Filter.by_property("in_stock").equal(False)
)
print("Đã delete tất cả in_stock=False")

new_count = collection.aggregate.over_all(total_count=True).total_count
print(f"Còn lại: {new_count} objects")


# ─────────────────────────────────────────────
# Dọn dẹp
# ─────────────────────────────────────────────
client.close()
print("\n✓ Đã đóng kết nối Weaviate")


print("\n" + "=" * 60)
print("TỔNG KẾT WEAVIATE")
print("=" * 60)
print("""
✓ Ưu điểm:
  - Schema rõ ràng, strongly typed
  - Hybrid Search tích hợp sẵn (vector + BM25)
  - Hỗ trợ nhiều vectorizer built-in
  - GraphQL API rất mạnh
  - Aggregate/thống kê tốt

✗ Nhược điểm:
  - API phức tạp nhất trong 4 loại
  - Tốn RAM hơn
  - near_text chỉ dùng được với built-in vectorizer

→ File tiếp theo: 05_rag_tong_hop.py
""")
