FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir runpod pillow torch diffusers transformers accelerate

COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
