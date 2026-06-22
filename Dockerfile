FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels — avoids pulling in the full CUDA stack (~1.5 GB)
RUN pip install --no-cache-dir \
    torch==2.12.1+cpu torchvision==0.27.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# torchreid is not on PyPI; install from git.
# Install its required inference deps first, then torchreid itself with
# --no-deps to skip heavy extras (matplotlib, tensorboard, scipy, h5py)
# that are only needed for training, not inference.
RUN pip install --no-cache-dir gdown six h5py scipy imageio opencv-python-headless
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/KaiyangZhou/deep-person-reid.git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TORCH_HOME controls where torch.hub caches downloaded repos and weights.
ENV TORCH_HOME=/app/torch_cache

# Pre-download OSNet x0.25 imagenet weights so the container starts cold.
# Patch builtins.input to suppress gdown's large-file confirmation prompt.
RUN python -c "import builtins; builtins.input = lambda *a: ''; from torchreid.utils import FeatureExtractor; FeatureExtractor(model_name='osnet_x0_25', device='cpu'); print('OSNet x0.25 weights cached.')"

COPY reid_service.py .

CMD ["python", "-u", "reid_service.py"]
