FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels — avoids pulling in the full CUDA stack (~1.5 GB)
RUN pip install --no-cache-dir \
    torch==2.12.1+cpu torchvision==0.27.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# gdown needed to fetch OSNet weights from Google Drive
RUN pip install --no-cache-dir gdown

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TORCH_HOME controls where osnet.py's init_pretrained_weights caches weights.
ENV TORCH_HOME=/app/torch_cache

# Fetch the self-contained OSNet implementation at a pinned commit (Apache-2.0).
# Only depends on torch — no torchreid package needed.
RUN curl -fsSL \
    "https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/c2d9b0e8251a7eb1f8417e703a1a1d90f7fca319/torchreid/models/osnet.py" \
    -o osnet.py

# Pre-download OSNet x0.25 imagenet weights so the container starts cold.
# Patch builtins.input to suppress gdown's large-file confirmation prompt.
RUN python -c "import builtins; builtins.input = lambda *a: ''; from osnet import osnet_x0_25, init_pretrained_weights; m = osnet_x0_25(num_classes=1000, pretrained=False); init_pretrained_weights(m, 'osnet_x0_25'); print('OSNet x0.25 weights cached.')"

COPY reid_service.py .

CMD ["python", "-u", "reid_service.py"]
