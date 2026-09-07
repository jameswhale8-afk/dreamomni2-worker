FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /

# Install the system-level tools required to build the AI engine
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
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
