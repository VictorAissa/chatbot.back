FROM python:3.10-slim
LABEL authors="victor"

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN ollama serve & \
    sleep 10 && \
    ollama pull gemma2:2b && \
    pkill ollama || true

EXPOSE 8000

VOLUME /app/data

CMD ollama serve & \
    sleep 5 && \
    cd /app && \
    python scripts/prepare_data.py && \
    python -m app.main