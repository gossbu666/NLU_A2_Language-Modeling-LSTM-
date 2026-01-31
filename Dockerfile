FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    procps \
    && apt-get clean \
    && ln -s /usr/bin/python3 /usr/bin/python

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_MODULE_LOADING=LAZY
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV VIRTUAL_ENV=/opt/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130



COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

RUN python -m nltk.downloader punkt stopwords wordnet

EXPOSE 8888 8501

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]