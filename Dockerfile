FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libaio-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install huggingface-hub && \
    huggingface-cli download \
        --resume-download \
        --local-dir-use-symlinks False \
        Na0s/Llama-3.2-3B-Medical-Chatbot-LoRA-FT \
        --local-dir /models

COPY handler.py .

CMD ["python", "-u", "/handler.py"]
