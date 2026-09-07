FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /

# Install system tools INCLUDING the Rust compiler (Critical for vLLM)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libaio-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Prepare Python environment
RUN python -m pip install --upgrade pip

# Install AI packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the medical model into the image
RUN pip install huggingface-hub && \
    huggingface-cli download \
        --resume-download \
        --local-dir-use-symlinks False \
        Na0s/Llama-3.2-3B-Medical-Chatbot-LoRA-FT \
        --local-dir /models

# Copy the handler script
COPY handler.py .

CMD ["python", "-u", "/handler.py"]
