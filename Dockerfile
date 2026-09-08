FROM runpod/pytorch:1.0.2-cu124-torch240-ubuntu2204
WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libaio-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Install all dependencies explicitly (bypasses requirements.txt build issues)
RUN pip install --no-cache-dir runpod==1.10.1 accelerate
RUN pip install --no-cache-dir vllm==0.6.3.post1 --extra-index-url https://download.pytorch.org/whl/cu124

RUN pip install huggingface-hub && \
    huggingface-cli download \
        --resume-download \
        --local-dir-use-symlinks False \
        Na0s/Llama-3.2-3B-Medical-Chatbot-LoRA-FT \
        --local-dir /models

COPY handler.py .

CMD ["python", "-u", "/handler.py"]
