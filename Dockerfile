FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface XDG_CACHE_HOME=/cache \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Keep upstream code and license notices, pinned independently of model weights.
RUN git clone https://github.com/Russell-Izadi-Bose/DiCoSe /opt/dicose \
    && git -C /opt/dicose checkout a1dc0a41ad2b1829674a60ea74e74edfd1509083 \
    && rm -rf /opt/dicose/.git \
    && git clone https://github.com/openmirlab/bs-roformer-infer /opt/bs-roformer-infer \
    && git -C /opt/bs-roformer-infer checkout 95c8de7b0b091d00f064d694bf916a21866c32f5 \
    && rm -rf /opt/bs-roformer-infer/.git

COPY scripts/docker-requirements.txt /opt/docker-requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/docker-requirements.txt \
    && python -m pip install --no-cache-dir --no-deps /opt/bs-roformer-infer

WORKDIR /opt/gaborformer
COPY pyproject.toml README.md LICENSE ./
COPY gaborformer ./gaborformer
COPY scripts/__init__.py scripts/listen.py scripts/competitors.py ./scripts/
RUN python -m pip install --no-cache-dir . \
    && python -m pip check \
    && mkdir -p /cache /work && chmod 1777 /cache /work

ENV PYTHONPATH=/opt/gaborformer
WORKDIR /work
USER 1000:1000
ENTRYPOINT ["python", "-m", "scripts.listen"]
CMD ["--help"]
