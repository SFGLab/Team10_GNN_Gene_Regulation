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

# install graphviz - latest stable version
RUN apt-get update && apt-get install -y graphviz graphviz-dev

# install python packages with NGC PyG 24.09 image versions
RUN pip install torch torch_geometric==2.6.0 triton==3.0.0 numba==0.59.0 requests==2.32.3 opencv-python==4.7.0.72 scipy==1.14.0 jupyterlab==4.2.5 networkx tensorflow scikit-learn clearml seaborn 

# install cugraph
RUN pip install cugraph-cu12 cugraph-pyg-cu12 --extra-index-url=https://pypi.nvidia.com

# workspace
RUN mkdir -p $WORK_DIR

WORKDIR $WORK_DIR

# copy data/scripts into container
COPY . .

# run pre-processing & training scripts
RUN python3 preprocessing_beeline.py && python3 train.py

CMD ["python3"]
