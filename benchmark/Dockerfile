FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV SRGAN_CHECKPOINT_PATH=/app/models/srgan.pth
ENV REALESRGAN_CHECKPOINT_PATH=/app/models/RealESRGAN_x4plus.pth

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/models && \
    python - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve

model_urls = {
    "srgan.pth": "https://raw.githubusercontent.com/mseitzer/srgan/master/resources/pretrained/srgan.pth",
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

for model_name, model_url in model_urls.items():
    model_path = Path("/app/models") / model_name

    if not model_path.exists():
        urlretrieve(model_url, model_path)
PY

COPY . ./

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
