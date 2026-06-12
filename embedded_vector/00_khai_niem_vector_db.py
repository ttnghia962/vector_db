"""
VECTOR DATABASE - KHÁI NIỆM CƠ BẢN
====================================

Vector Database lưu trữ dữ liệu dưới dạng VECTOR (mảng số thực)
thay vì text, và tìm kiếm bằng ĐỘ TƯƠNG TỰ (similarity) thay vì
so khớp từ khóa chính xác.

LUỒNG XỬ LÝ:
  Text/Image/Audio  →  Embedding Model  →  Vector  →  Vector DB
  "con mèo"         →  [0.12, -0.45, ...]            →  lưu vào DB

  Câu hỏi: "mèo nhà"  →  vector hỏi  →  tìm vector gần nhất  →  kết quả
"""

import numpy as np

# ─────────────────────────────────────────────
# 1. Vector là gì?
# ─────────────────────────────────────────────
print("=" * 60)
print("1. VÍ DỤ VỀ VECTOR EMBEDDING")
print("=" * 60)

# Giả lập embedding đơn giản (thực tế dùng model như BERT, OpenAI...)
# Mỗi từ/câu được chuyển thành một mảng số
embeddings = {
    "con chó":     np.array([0.9, 0.1, 0.8, 0.2]),
    "con mèo":     np.array([0.8, 0.2, 0.7, 0.3]),
    "xe ô tô":     np.array([0.1, 0.9, 0.2, 0.8]),
    "xe máy":      np.array([0.2, 0.8, 0.3, 0.7]),
}

print("Các vector embedding:")
for word, vec in embeddings.items():
    print(f"  '{word}': {vec}")


# ─────────────────────────────────────────────
# 2. Cosine Similarity - đo độ tương đồng
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. COSINE SIMILARITY")
print("=" * 60)

def cosine_similarity(vec_a, vec_b):
    """Giá trị từ -1 đến 1, càng gần 1 càng giống nhau"""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)


query = "thú cưng"
query_vec = np.array([0.85, 0.15, 0.75, 0.25])  # gần với "chó" và "mèo"

print(f"\nTìm kiếm: '{query}' → vector: {query_vec}")
print("\nĐộ tương đồng với từng mục:")

results = []
for word, vec in embeddings.items():
    score = cosine_similarity(query_vec, vec)
    results.append((word, score))
    print(f"  '{word}': {score:.4f}")

results.sort(key=lambda x: x[1], reverse=True)
print(f"\n→ Kết quả gần nhất: '{results[0][0]}' (score: {results[0][1]:.4f})")


# ─────────────────────────────────────────────
# 3. So sánh 4 Vector DB
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SO SÁNH 4 VECTOR DATABASE")
print("=" * 60)

comparison = """
┌─────────────┬──────────────┬────────────┬──────────────┬─────────────┐
│             │  ChromaDB    │  Pinecone  │   Qdrant     │  Weaviate   │
├─────────────┼──────────────┼────────────┼──────────────┼─────────────┤
│ Chạy ở đâu │ Local/Cloud  │ Cloud only │ Local/Cloud  │ Local/Cloud │
│ Dễ dùng    │ ★★★★★        │ ★★★★☆     │ ★★★★☆       │ ★★★☆☆      │
│ Filter      │ Cơ bản       │ Tốt        │ Rất tốt      │ Rất tốt     │
│ GraphQL     │ Không        │ Không      │ Không        │ Có          │
│ Multi-modal │ Không        │ Không      │ Có           │ Có          │
│ Free tier   │ Hoàn toàn    │ Hạn chế    │ Hoàn toàn    │ Hạn chế     │
│ Use case    │ Prototype    │ Production │ Production   │ Production  │
│             │ RAG nhỏ      │ Scale lớn  │ Filtering    │ Semantic    │
└─────────────┴──────────────┴────────────┴──────────────┴─────────────┘

KHUYẾN NGHỊ:
  - Mới học / prototype  → ChromaDB (đơn giản nhất, không cần server)
  - Cần filter phức tạp  → Qdrant
  - Cần GraphQL / schema  → Weaviate
  - Cần managed cloud    → Pinecone
"""
print(comparison)


# ─────────────────────────────────────────────
# 4. Các khái niệm chung
# ─────────────────────────────────────────────
print("=" * 60)
print("4. THUẬT NGỮ CHUNG")
print("=" * 60)

concepts = """
Collection / Index / Namespace:
  → Tương đương "bảng" trong SQL, nơi lưu các vector

Document / Record:
  → Một bản ghi = vector + metadata + id

Metadata:
  → Dữ liệu kèm theo vector, dùng để filter (vd: category, date...)

Upsert:
  → Insert nếu chưa có, Update nếu đã có (theo id)

k-NN (k Nearest Neighbors):
  → Tìm k vector gần nhất với vector truy vấn

ANN (Approximate Nearest Neighbors):
  → Phiên bản xấp xỉ của kNN, nhanh hơn nhiều (HNSW, IVF...)

HNSW (Hierarchical Navigable Small World):
  → Thuật toán index phổ biến nhất, cân bằng tốc độ/độ chính xác
"""
print(concepts)

print("→ File tiếp theo: 01_chromadb_demo.py")
