FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch trước — giảm image size từ ~6GB xuống ~2GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY embedded_vector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model vào image (không cần internet khi chạy)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["bash"]
