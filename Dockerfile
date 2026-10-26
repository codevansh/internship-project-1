FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Tesseract is used for scanned PDF pages. English is the only OCR language
# currently requested by pytesseract (its default is `eng`). libgomp is used
# by CPU builds of PyTorch and FAISS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# The inference code needs the label map, but downloads its configured
# Hugging Face models when first used. Do not bake training checkpoints or
# datasets into this image.
COPY src/ ./src/
COPY data/processed/clause_label_mapping.json ./data/processed/clause_label_mapping.json

EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
