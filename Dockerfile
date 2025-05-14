FROM nvcr.io/nvidia/cuda-dl-base:24.09-cuda12.6-devel-ubuntu22.04

# Based on NGC PyG 24.09 image:
# https://docs.nvidia.com/deeplearning/frameworks/pyg-release-notes/rel-24-09.html#rel-24-09

# reduce package overhead
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORK_DIR="/opt/GNN_GRN"

# install pip
RUN apt-get update && apt-get install -y python3-pip

# install PyTorch - latest stable version
RUN pip install torch torchvision torchaudio

# install graphviz - latest stable version
RUN apt-get install -y graphviz graphviz-dev
RUN pip install pygraphviz

# install python packages with NGC PyG 24.09 image versions
RUN pip install torch_geometric==2.6.0
RUN pip install triton==3.0.0 numba==0.59.0 requests==2.32.3 opencv-python==4.7.0.72 scipy==1.14.0 jupyterlab==4.2.5

# install cugraph
RUN pip install cugraph-cu12 cugraph-pyg-cu12 --extra-index-url=https://pypi.nvidia.com

# user (for better security, don't run as root)
RUN useradd -m -s /bin/bash pymonk && \
    echo "pymonk:password" | chpasswd && \
    usermod -aG sudo pymonk
USER pymonk

# workspace
RUN mkdir -p $WORK_DIR && \
chown -R pymonk $WORK_DIR

WORKDIR $WORK_DIR

COPY . .



