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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MobileNetV3-Small weights into the image so the container
# starts immediately without a network fetch on first run.
RUN python -c "import torchvision.models as m; m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)"

COPY reid_service.py .

CMD ["python", "-u", "reid_service.py"]
