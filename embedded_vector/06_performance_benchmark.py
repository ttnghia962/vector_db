"""
PERFORMANCE BENCHMARK - ChromaDB với dữ liệu thực tế
======================================================
Dữ liệu: POLK Vehicle YMME List (~310k xe ô tô)
Đo: thời gian embed, insert, query ở các quy mô khác nhau

Cài thêm nếu chưa có:
  pip install chromadb sentence-transformers openpyxl psutil
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import tracemalloc
import gc
import psutil
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

EXCEL_PATH = r'E:\02_[REPO]\FORLEARNING\embedded_vector\POLK Vehicle YMME List_Oct182025.xlsx'
# Các mức dữ liệu cần test
TEST_SCALES = [500, 2_000, 10_000, 50_000]

# Queries thực tế để test tìm kiếm
TEST_QUERIES = [
    "Ford pickup truck diesel engine 2008",
    "compact sedan gas automatic transmission",
    "heavy duty truck V8 engine",
    "hybrid electric vehicle Toyota",
    "sports car manual transmission high performance",
]


# ─────────────────────────────────────────────
# Utility: đo RAM của process hiện tại
# ─────────────────────────────────────────────
def get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_separator(title=""):
    line = "=" * 65
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


# ─────────────────────────────────────────────
# 1. Load dữ liệu từ Excel
# ─────────────────────────────────────────────
print_separator("BƯỚC 1: ĐỌC DỮ LIỆU TỪ EXCEL")

t0 = time.perf_counter()
df = pd.read_excel(EXCEL_PATH, dtype=str)
df = df.fillna("")
load_time = time.perf_counter() - t0

print(f"  File       : {EXCEL_PATH}")
print(f"  Tổng dòng  : {len(df):,}")
print(f"  Cột        : {list(df.columns[:8])} ...")
print(f"  Thời gian  : {load_time:.2f}s")

# Tạo cột text mô tả xe (dùng làm document để embed)
df["doc"] = (
    df["Year"].str.strip() + " " +
    df["Make"].str.strip() + " " +
    df["Model"].str.strip() + " " +
    df["Trimlevel"].str.strip() + " " +
    df["EngineType"].str.strip() + " " +
    df["FuelType"].str.strip() + " " +
    df["BodyStyleDesc"].str.strip()
).str.strip()

print(f"\n  Ví dụ document:")
for i in range(3):
    print(f"    [{i+1}] {df['doc'].iloc[i]}")


# ─────────────────────────────────────────────
# 2. Load Embedding Model (1 lần duy nhất)
# ─────────────────────────────────────────────
print_separator("BƯỚC 2: LOAD EMBEDDING MODEL")

t0 = time.perf_counter()
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
# Warm-up: encode 1 câu để model load xong hoàn toàn
_ = ef(["warm up"])
model_load_time = time.perf_counter() - t0

print(f"  Model      : all-MiniLM-L6-v2 (384 chiều)")
print(f"  Thời gian  : {model_load_time:.2f}s")
print(f"  RAM sau load: {get_ram_mb():.0f} MB")


# ─────────────────────────────────────────────
# 3. Benchmark theo từng quy mô
# ─────────────────────────────────────────────
print_separator("BƯỚC 3: BENCHMARK THEO QUY MÔ")

results_table = []  # lưu kết quả để in bảng tổng kết

for scale in TEST_SCALES:
    print(f"\n{'─'*65}")
    print(f"  QUY MÔ: {scale:,} dòng")
    print(f"{'─'*65}")

    # Lấy subset dữ liệu
    subset = df.head(scale)
    docs      = subset["doc"].tolist()
    ids       = [f"xe_{i}" for i in range(scale)]
    metadatas = subset[["Year", "Make", "Model", "FuelType", "BodyStyleDesc"]].to_dict("records")

    ram_before = get_ram_mb()

    # ── 3a. Thời gian tạo Embedding ──────────────
    EMBED_BATCH = 512   # ChromaDB gọi model theo batch
    t0 = time.perf_counter()
    # Tạo embedding thủ công để đo riêng thời gian
    all_embeddings = []
    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i:i + EMBED_BATCH]
        all_embeddings.extend(ef(batch))
    embed_time = time.perf_counter() - t0
    embed_per_sec = scale / embed_time

    print(f"  [EMBED]  {embed_time:6.2f}s  →  {embed_per_sec:,.0f} docs/s")

    # ── 3b. Thời gian Insert vào ChromaDB ────────
    client = chromadb.Client()  # in-memory mỗi lần để không bị ảnh hưởng
    col = client.create_collection(
        name=f"xe_{scale}",
        metadata={"hnsw:space": "cosine"}
    )

    UPSERT_BATCH = 1000
    t0 = time.perf_counter()
    for i in range(0, scale, UPSERT_BATCH):
        col.add(
            ids=ids[i:i + UPSERT_BATCH],
            embeddings=all_embeddings[i:i + UPSERT_BATCH],
            documents=docs[i:i + UPSERT_BATCH],
            metadatas=metadatas[i:i + UPSERT_BATCH],
        )
    insert_time = time.perf_counter() - t0
    insert_per_sec = scale / insert_time

    print(f"  [INSERT] {insert_time:6.2f}s  →  {insert_per_sec:,.0f} docs/s")

    # ── 3c. Thời gian Query đơn lẻ ───────────────
    query_times = []
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        col.query(query_texts=[q], n_results=5)
        query_times.append(time.perf_counter() - t0)

    avg_query_ms = (sum(query_times) / len(query_times)) * 1000
    min_query_ms = min(query_times) * 1000
    max_query_ms = max(query_times) * 1000

    print(f"  [QUERY]  avg={avg_query_ms:.1f}ms  min={min_query_ms:.1f}ms  max={max_query_ms:.1f}ms  ({len(TEST_QUERIES)} queries)")

    # ── 3d. Thời gian Query có Filter ────────────
    filter_times = []
    test_filters = [
        {"Make": "FORD"},
        {"FuelType": "Diesel"},
        {"FuelType": "Gas"},
    ]
    for flt in test_filters:
        t0 = time.perf_counter()
        col.query(
            query_texts=["pickup truck heavy duty"],
            n_results=5,
            where=flt
        )
        filter_times.append(time.perf_counter() - t0)

    avg_filter_ms = (sum(filter_times) / len(filter_times)) * 1000
    print(f"  [FILTER] avg={avg_filter_ms:.1f}ms  ({len(test_filters)} queries với where=...)")

    # ── 3e. RAM ───────────────────────────────────
    ram_after = get_ram_mb()
    ram_delta = ram_after - ram_before
    print(f"  [RAM]    +{ram_delta:.0f} MB  (tổng: {ram_after:.0f} MB)")

    # Lưu kết quả
    results_table.append({
        "scale": scale,
        "embed_time": embed_time,
        "embed_per_sec": embed_per_sec,
        "insert_time": insert_time,
        "insert_per_sec": insert_per_sec,
        "avg_query_ms": avg_query_ms,
        "avg_filter_ms": avg_filter_ms,
        "ram_delta_mb": ram_delta,
    })

    # Dọn dẹp để không ảnh hưởng lần sau
    del col, all_embeddings, docs, ids, metadatas, subset
    gc.collect()


# ─────────────────────────────────────────────
# 4. Bảng tổng kết
# ─────────────────────────────────────────────
print_separator("BƯỚC 4: BẢNG TỔNG KẾT")

print(f"\n  {'Scale':>8}  {'Embed(s)':>9}  {'Embed/s':>9}  {'Insert(s)':>10}  {'Insert/s':>9}  {'Query(ms)':>10}  {'Filter(ms)':>11}  {'RAM(MB)':>8}")
print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*8}")

for r in results_table:
    print(
        f"  {r['scale']:>8,}"
        f"  {r['embed_time']:>9.2f}"
        f"  {r['embed_per_sec']:>9,.0f}"
        f"  {r['insert_time']:>10.2f}"
        f"  {r['insert_per_sec']:>9,.0f}"
        f"  {r['avg_query_ms']:>10.1f}"
        f"  {r['avg_filter_ms']:>11.1f}"
        f"  {r['ram_delta_mb']:>8.0f}"
    )


# ─────────────────────────────────────────────
# 5. Phân tích tốc độ tăng (scaling factor)
# ─────────────────────────────────────────────
print_separator("BƯỚC 5: PHÂN TÍCH SCALING")

base = results_table[0]
print(f"\n  So với baseline ({base['scale']:,} dòng):\n")
print(f"  {'Scale':>8}  {'Data×':>7}  {'Embed×':>8}  {'Insert×':>9}  {'Query×':>8}")
print(f"  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}")

for r in results_table:
    data_x   = r["scale"] / base["scale"]
    embed_x  = r["embed_time"] / base["embed_time"]
    insert_x = r["insert_time"] / base["insert_time"]
    query_x  = r["avg_query_ms"] / base["avg_query_ms"]
    print(
        f"  {r['scale']:>8,}"
        f"  {data_x:>7.0f}x"
        f"  {embed_x:>8.1f}x"
        f"  {insert_x:>9.1f}x"
        f"  {query_x:>8.1f}x"
    )

print(f"""
  Nhận xét:
  → Embed  : tuyến tính O(n) — 10× dữ liệu ≈ 10× thời gian
  → Insert : tuyến tính O(n) — phụ thuộc I/O
  → Query  : gần như O(log n) nhờ HNSW — tăng rất ít dù dữ liệu lớn
             Đây là lý do vector DB dùng HNSW thay vì linear scan!
""")


# ─────────────────────────────────────────────
# 6. Demo kết quả tìm kiếm thực tế
# ─────────────────────────────────────────────
print_separator("BƯỚC 6: XEM KẾT QUẢ TÌM KIẾM THỰC TẾ (10k dòng)")

subset_demo = df.head(10_000)
docs_demo   = subset_demo["doc"].tolist()
ids_demo    = [f"xe_{i}" for i in range(10_000)]
meta_demo   = subset_demo[["Year","Make","Model","FuelType","BodyStyleDesc"]].to_dict("records")

client_demo = chromadb.Client()
col_demo = client_demo.create_collection("demo", metadata={"hnsw:space": "cosine"})

print("\n  Đang insert 10,000 xe vào ChromaDB...", end="", flush=True)
embeddings_demo = []
for i in range(0, 10_000, 512):
    embeddings_demo.extend(ef(docs_demo[i:i+512]))
for i in range(0, 10_000, 1000):
    col_demo.add(
        ids=ids_demo[i:i+1000],
        embeddings=embeddings_demo[i:i+1000],
        documents=docs_demo[i:i+1000],
        metadatas=meta_demo[i:i+1000],
    )
print(" done")

demo_queries = [
    "Ford diesel pickup truck 2008",
    "Toyota hybrid sedan automatic",
    "heavy duty commercial truck V8",
]

for q in demo_queries:
    t0 = time.perf_counter()
    res = col_demo.query(query_texts=[q], n_results=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Query ({elapsed_ms:.1f}ms): \"{q}\"")
    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        score = 1 - dist
        print(f"    [{score:.3f}] {meta['Year']} {meta['Make']} {meta['Model']} | {meta['FuelType']} | {meta['BodyStyleDesc']}")

print("\n" + "=" * 65)
print("  BENCHMARK HOÀN THÀNH")
print("=" * 65)
