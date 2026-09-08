FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libaio-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install core deps first (bypasses requirements.txt issues entirely)
RUN pip install --no-cache-dir runpod==1.10.1 accelerate

# Install vLLM separately so we catch any error
RUN pip install --no-cache-dir vllm

# Download model during build (done once, cached in Docker layer)
RUN pip install huggingface-hub && \
    huggingface-cli download \
        --resume-download \
        --local-dir-use-symlinks False \
        Na0s/Llama-3.2-3B-Medical-Chatbot-LoRA-FT \
        --local-dir /models

COPY handler.py .

CMD ["python", "-u", "/handler.py"]
