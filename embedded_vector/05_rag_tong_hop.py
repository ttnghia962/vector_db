"""
RAG (Retrieval Augmented Generation) - Ứng dụng thực tế
=========================================================
Kết hợp Vector DB + LLM để trả lời câu hỏi từ tài liệu riêng.

Luồng RAG:
  [Tài liệu] → chunk → embed → lưu vào VectorDB
  [Câu hỏi]  → embed → tìm chunk liên quan → ghép vào prompt → LLM → trả lời

Demo này dùng ChromaDB (đơn giản nhất) + OpenAI GPT
Cài thêm: pip install openai
"""

import os
import chromadb
from chromadb.utils import embedding_functions
# from openai import OpenAI  # uncomment khi có API key


# ─────────────────────────────────────────────
# 1. Chuẩn bị tài liệu (Knowledge Base)
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CHUẨN BỊ TÀI LIỆU")
print("=" * 60)

# Dữ liệu mô phỏng: chính sách công ty
tai_lieu = [
    {
        "id": "policy_001",
        "text": "Chính sách nghỉ phép: Nhân viên được nghỉ 12 ngày phép năm. "
                "Phép năm được tính từ ngày 1 tháng 1 hàng năm. "
                "Phép chưa dùng hết có thể chuyển sang năm sau tối đa 5 ngày.",
        "source": "HR Policy 2024",
        "category": "hr"
    },
    {
        "id": "policy_002",
        "text": "Chính sách làm việc từ xa (Remote): Nhân viên được làm việc từ xa "
                "tối đa 3 ngày/tuần. Phải online trên Slack trong giờ hành chính 9h-18h. "
                "Cần thông báo trước 24 giờ khi muốn làm remote.",
        "source": "HR Policy 2024",
        "category": "hr"
    },
    {
        "id": "policy_003",
        "text": "Quy trình onboarding nhân viên mới: Tuần 1 học về sản phẩm và quy trình. "
                "Tuần 2-4 làm việc cùng mentor. "
                "Sau 3 tháng có đánh giá thử việc. Lương thử việc bằng 85% lương chính thức.",
        "source": "HR Policy 2024",
        "category": "hr"
    },
    {
        "id": "tech_001",
        "text": "Stack công nghệ backend: Python FastAPI, PostgreSQL, Redis, Docker. "
                "Deploy trên AWS với ECS Fargate. CI/CD dùng GitHub Actions. "
                "Code review bắt buộc trước khi merge vào main branch.",
        "source": "Tech Wiki",
        "category": "tech"
    },
    {
        "id": "tech_002",
        "text": "Stack công nghệ frontend: React TypeScript, TailwindCSS, Vite. "
                "State management dùng Zustand. Testing với Vitest và Playwright. "
                "Deploy trên Vercel.",
        "source": "Tech Wiki",
        "category": "tech"
    },
    {
        "id": "product_001",
        "text": "Sản phẩm chính: Nền tảng quản lý dự án B2B. "
                "Có 3 tier: Starter (miễn phí, 5 users), Pro ($99/tháng, 50 users), "
                "Enterprise (tùy chỉnh, unlimited users). "
                "Tích hợp với Slack, Jira, GitHub.",
        "source": "Product Doc",
        "category": "product"
    },
    {
        "id": "product_002",
        "text": "Roadmap Q1 2025: Ra mắt tính năng AI Assistant giúp tự động tóm tắt task. "
                "Tích hợp Google Calendar. Cải thiện mobile app. "
                "Target: tăng MRR 30%, giảm churn rate xuống dưới 5%.",
        "source": "Product Doc",
        "category": "product"
    },
]

print(f"Đã chuẩn bị {len(tai_lieu)} tài liệu")


# ─────────────────────────────────────────────
# 2. Chunk Documents (chia nhỏ tài liệu dài)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CHUNKING TÀI LIỆU DÀI")
print("=" * 60)

def chunk_text(text, chunk_size=200, overlap=50):
    """
    Chia text dài thành các đoạn nhỏ có overlap.
    Overlap giúp giữ ngữ cảnh ở ranh giới giữa các chunk.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# Demo chunking với text dài
long_text = """
Đây là một tài liệu rất dài về lịch sử phát triển của trí tuệ nhân tạo.
Bắt đầu từ những năm 1950 với Alan Turing và bài kiểm tra Turing nổi tiếng.
Qua các thập kỷ phát triển, AI đã trải qua nhiều làn sóng và mùa đông.
Deep Learning bùng nổ vào năm 2012 với AlexNet thắng ImageNet.
Năm 2017 Transformer được giới thiệu, thay đổi hoàn toàn NLP.
GPT-3 ra đời năm 2020 với 175 tỷ tham số gây chấn động.
ChatGPT ra mắt tháng 11/2022 đạt 100 triệu người dùng trong 2 tháng.
"""
chunks = chunk_text(long_text, chunk_size=30, overlap=5)
print(f"Text gốc: {len(long_text.split())} từ → {len(chunks)} chunks")
for i, c in enumerate(chunks):
    print(f"  Chunk {i}: {c[:60]}...")


# ─────────────────────────────────────────────
# 3. Lưu vào Vector DB
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. LƯU VÀO VECTOR DB (ChromaDB)")
print("=" * 60)

ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    chroma_client.delete_collection("knowledge_base")
except Exception:
    pass

kb = chroma_client.create_collection(
    name="knowledge_base",
    embedding_function=ef,
    metadata={"hnsw:space": "cosine"}
)

# Thêm tài liệu
kb.add(
    ids=[doc["id"] for doc in tai_lieu],
    documents=[doc["text"] for doc in tai_lieu],
    metadatas=[
        {"source": doc["source"], "category": doc["category"]}
        for doc in tai_lieu
    ]
)
print(f"✓ Đã lưu {kb.count()} tài liệu vào knowledge base")


# ─────────────────────────────────────────────
# 4. Retriever - Tìm tài liệu liên quan
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. RETRIEVER")
print("=" * 60)

def retrieve(query, n_results=3, category=None):
    """Tìm các tài liệu liên quan nhất đến câu hỏi"""
    where = {"category": category} if category else None
    results = kb.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )

    context_docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        score = 1 - dist
        context_docs.append({
            "text": doc,
            "source": meta["source"],
            "score": score
        })
    return context_docs


# Test retriever
cau_hoi = "Tôi được nghỉ bao nhiêu ngày phép?"
docs = retrieve(cau_hoi)
print(f"\nCâu hỏi: '{cau_hoi}'")
print("Tài liệu liên quan:")
for d in docs:
    print(f"  [{d['score']:.3f}] ({d['source']}) {d['text'][:80]}...")


# ─────────────────────────────────────────────
# 5. RAG - Kết hợp Retriever + Generator
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. RAG PIPELINE")
print("=" * 60)

def build_prompt(question: str, context_docs: list) -> str:
    """Xây dựng prompt kết hợp câu hỏi + context từ vector DB"""
    context = "\n\n".join([
        f"[Tài liệu {i+1} - {d['source']}]:\n{d['text']}"
        for i, d in enumerate(context_docs)
    ])

    prompt = f"""Bạn là trợ lý AI của công ty. Hãy trả lời câu hỏi dựa trên tài liệu được cung cấp.
Chỉ trả lời dựa trên thông tin có trong tài liệu. Nếu không có thông tin, hãy nói "Tôi không tìm thấy thông tin về vấn đề này."

=== TÀI LIỆU THAM KHẢO ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ==="""
    return prompt


def rag_answer(question: str, use_llm: bool = False) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant docs
    2. Build prompt
    3. Generate answer with LLM (hoặc mock nếu không có API key)
    """
    # Step 1: Retrieve
    context_docs = retrieve(question, n_results=3)

    # Step 2: Build prompt
    prompt = build_prompt(question, context_docs)

    if use_llm:
        # Step 3: Generate (cần OpenAI API key)
        # openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response = openai_client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        return "[Cần OPENAI_API_KEY để dùng tính năng này]"
    else:
        # Mock: chỉ hiển thị prompt (để học cách RAG hoạt động)
        return f"[MOCK - Prompt đã sẵn sàng]\n{prompt[:400]}..."


# Test RAG
questions = [
    "Tôi được nghỉ bao nhiêu ngày phép trong năm?",
    "Công ty dùng công nghệ gì cho backend?",
    "Chi phí plan Pro là bao nhiêu?",
    "Lương thử việc là bao nhiêu phần trăm?",
]

for q in questions:
    print(f"\nQ: {q}")
    docs_found = retrieve(q, n_results=2)
    print(f"→ Tài liệu liên quan nhất: [{docs_found[0]['score']:.3f}] {docs_found[0]['text'][:80]}...")


# ─────────────────────────────────────────────
# 6. Hiển thị full RAG example
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FULL RAG PROMPT EXAMPLE")
print("=" * 60)

q = "Nhân viên mới được hưởng lương bao nhiêu trong thời gian thử việc?"
context = retrieve(q, n_results=2)
prompt = build_prompt(q, context)
print(prompt)


print("\n" + "=" * 60)
print("TỔNG KẾT RAG")
print("=" * 60)
print("""
RAG Pipeline hoàn chỉnh:

  1. INDEXING (làm offline, 1 lần):
     Documents → Chunk → Embed → VectorDB

  2. RETRIEVAL (mỗi khi có query):
     Query → Embed → Similarity Search → Top-K Docs

  3. GENERATION (mỗi khi có query):
     Query + Top-K Docs → Prompt → LLM → Answer

Best practices:
  - Chunk size: 200-500 tokens, overlap 10-20%
  - Top-K: thường 3-5 (nhiều hơn → nhiều nhiễu)
  - Reranking: dùng cross-encoder để rerank sau retrieval
  - Hybrid: kết hợp dense + sparse (BM25) cho tốt nhất
  - Score threshold: loại bỏ kết quả không đủ liên quan
""")
