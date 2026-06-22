FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels — avoids pulling in the full CUDA stack (~1.5 GB)
RUN pip install --no-cache-dir \
    torch==2.12.1+cpu torchvision==0.27.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# gdown is needed by torchreid's hub loader to fetch weights from Google Drive
RUN pip install --no-cache-dir gdown

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TORCH_HOME controls where torch.hub caches downloaded repos and weights.
# Setting it to a fixed path inside the image ensures the pre-download below
# is found at the same location when the container starts.
ENV TORCH_HOME=/app/torch_cache

# Pre-download OSNet x0.25 (torchreid hub) so the container starts without
# a network fetch. Weights are ~4 MB; the torchreid repo clone is ~15 MB.
# Patch builtins.input before the hub load: gdown calls input() to confirm
# Google Drive large-file downloads, which causes EOFError in non-TTY builds.
RUN python -c "import builtins; builtins.input = lambda *a: ''; import torch; torch.hub.load('KaiyangZhou/deep-person-reid', 'osnet_x0_25', pretrained=True, verbose=False); print('OSNet x0.25 weights cached.')"

COPY reid_service.py .

CMD ["python", "-u", "reid_service.py"]
