FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
WORKDIR /
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY handler.py .
EXPOSE 8080
CMD ["python", "-u", "/handler.py"]
