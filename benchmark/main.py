import base64
import io
import os
import time

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from realesrgan import load_realesrgan_generator
from srgan import load_srgan_generator


checkpoint_path = os.environ.get("SRGAN_CHECKPOINT_PATH", "/app/models/srgan.pth")
realesrgan_checkpoint_path = os.environ.get("REALESRGAN_CHECKPOINT_PATH", "/app/models/RealESRGAN_x4plus.pth")
preferred_device = os.environ.get("SRGAN_DEVICE", "cuda")
sr_model_max_input_pixels = int(os.environ.get("SR_MODEL_MAX_INPUT_PIXELS", "65536"))
device = torch.device("cuda" if preferred_device == "cuda" and torch.cuda.is_available() else "cpu")
srgan_generator = load_srgan_generator(checkpoint_path, device)
realesrgan_generator = load_realesrgan_generator(realesrgan_checkpoint_path, device)

app = FastAPI(title="SR Benchmark Inference API")


@app.get("/api/srgan/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "service": "sr-benchmark-api",
        "models": ["mseitzer/srgan", "Real-ESRGAN x4plus"],
        "scaleFactor": 4,
        "maxInputPixels": sr_model_max_input_pixels,
        "device": str(device),
        "cudaAvailable": torch.cuda.is_available(),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/api/srgan/upscale")
async def upscale(image: UploadFile = File(...)):
    return await upscale_with_model(
        image,
        "SRGAN",
        "mseitzer/srgan pretrained COCO checkpoint",
        "SRGAN x4 결과 이미지를 생성했습니다.",
        run_srgan_inference)


@app.post("/api/realesrgan/upscale")
async def upscale_realesrgan(image: UploadFile = File(...)):
    return await upscale_with_model(
        image,
        "Real-ESRGAN",
        "RealESRGAN_x4plus official pretrained checkpoint",
        "Real-ESRGAN x4 결과 이미지를 생성했습니다.",
        run_realesrgan_inference)


async def upscale_with_model(image, model_name, model_source, output_message, inference_function):
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="image 파일이 비어 있습니다.")

    try:
        low_resolution_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.") from exception

    validate_sr_model_input_size(low_resolution_image, model_name)

    started_at = time.perf_counter()
    try:
        output_image = inference_function(low_resolution_image)
    except torch.OutOfMemoryError as exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=f"{model_name} GPU 메모리가 부족합니다. 더 작은 이미지로 테스트해 주세요.") from exception

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)

    output_stream = io.BytesIO()
    output_image.save(output_stream, format="PNG")
    output_bytes = output_stream.getvalue()
    output_base64 = base64.b64encode(output_bytes).decode("ascii")

    return {
        "success": True,
        "model": {
            "name": model_name,
            "source": model_source,
            "scaleFactor": 4,
            "device": str(device),
        },
        "input": {
            "fileName": image.filename,
            "contentType": image.content_type,
            "width": low_resolution_image.width,
            "height": low_resolution_image.height,
        },
        "output": {
            "targetWidth": output_image.width,
            "targetHeight": output_image.height,
            "resultImage": f"data:image/png;base64,{output_base64}",
            "elapsedMs": elapsed_ms,
            "message": output_message,
        },
    }


def validate_sr_model_input_size(low_resolution_image, model_name):
    input_pixel_count = low_resolution_image.width * low_resolution_image.height

    if input_pixel_count <= sr_model_max_input_pixels:
        return

    max_square_side = int(sr_model_max_input_pixels ** 0.5)

    raise HTTPException(
        status_code=413,
        detail=(
            f"{model_name} 입력 이미지가 너무 큽니다. "
            f"{max_square_side}x{max_square_side} 이하 또는 {sr_model_max_input_pixels} pixels 이하 이미지로 테스트해 주세요."))


def run_srgan_inference(low_resolution_image):
    input_tensor = convert_image_to_tensor(low_resolution_image).to(device)

    with torch.inference_mode():
        output_tensor = srgan_generator(input_tensor)

    output_tensor = normalize_model_output(output_tensor)

    return convert_tensor_to_image(output_tensor)


def run_realesrgan_inference(low_resolution_image):
    input_tensor = convert_rgb_image_to_bgr_tensor(low_resolution_image).to(device)

    with torch.inference_mode():
        output_tensor = realesrgan_generator(input_tensor)

    output_tensor = output_tensor.detach().float().cpu().clamp(0.0, 1.0)

    return convert_bgr_tensor_to_rgb_image(output_tensor)


def convert_image_to_tensor(image):
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return image_tensor


def convert_rgb_image_to_bgr_tensor(image):
    image_array = np.asarray(image, dtype=np.float32)[:, :, ::-1] / 255.0
    image_array = np.ascontiguousarray(image_array.transpose(2, 0, 1))

    return torch.from_numpy(image_array).unsqueeze(0)


def normalize_model_output(output_tensor):
    output_tensor = output_tensor.detach().float().cpu()
    output_tensor = (output_tensor + 1.0) / 2.0

    return output_tensor.clamp(0.0, 1.0)


def convert_tensor_to_image(output_tensor):
    image_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
    image_array = (image_tensor.numpy() * 255.0 + 0.5).astype(np.uint8)

    return Image.fromarray(image_array, mode="RGB")


def convert_bgr_tensor_to_rgb_image(output_tensor):
    image_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
    image_array = (image_tensor.numpy() * 255.0 + 0.5).astype(np.uint8)
    image_array = np.ascontiguousarray(image_array[:, :, ::-1])

    return Image.fromarray(image_array, mode="RGB")
