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
