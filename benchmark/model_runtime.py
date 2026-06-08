import base64
import io
import os
import time

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image

from realesrgan import load_realesrgan_generator
from srgan import load_srgan_generator


SCALE_FACTOR = 4
CHECKPOINT_PATH = os.environ.get("SRGAN_CHECKPOINT_PATH", "/app/models/srgan.pth")
REALESRGAN_CHECKPOINT_PATH = os.environ.get("REALESRGAN_CHECKPOINT_PATH", "/app/models/RealESRGAN_x4plus.pth")
PREFERRED_DEVICE = os.environ.get("SRGAN_DEVICE", "cuda")
SR_MODEL_MAX_INPUT_PIXELS = int(os.environ.get("SR_MODEL_MAX_INPUT_PIXELS", "65536"))
DEVICE = torch.device("cuda" if PREFERRED_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
SRGAN_GENERATOR = load_srgan_generator(CHECKPOINT_PATH, DEVICE)
REALESRGAN_GENERATOR = load_realesrgan_generator(REALESRGAN_CHECKPOINT_PATH, DEVICE)


def create_health_response():
    return {
        "success": True,
        "status": "ok",
        "service": "sr-benchmark-api",
        "models": ["mseitzer/srgan", "Real-ESRGAN x4plus"],
        "scaleFactor": SCALE_FACTOR,
        "maxInputPixels": SR_MODEL_MAX_INPUT_PIXELS,
        "device": str(DEVICE),
        "cudaAvailable": torch.cuda.is_available(),
        "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


async def upscale_with_model(image_file, model_name, model_source, output_message, inference_function):
    image_bytes = await image_file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="image 파일이 비어 있습니다.")

    try:
        low_resolution_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.") from exception

    validate_model_input_size(low_resolution_image, model_name)

    started_at = time.perf_counter()
    try:
        output_image = inference_function(low_resolution_image)
    except torch.OutOfMemoryError as exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(status_code=507, detail=f"{model_name} GPU 메모리가 부족합니다. 더 작은 이미지로 테스트해 주세요.") from exception

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    output_bytes = encode_png(output_image)
    output_base64 = base64.b64encode(output_bytes).decode("ascii")

    return {
        "success": True,
        "model": {
            "name": model_name,
            "source": model_source,
            "scaleFactor": SCALE_FACTOR,
            "device": str(DEVICE),
        },
        "input": {
            "fileName": image_file.filename,
            "contentType": image_file.content_type,
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


def validate_model_input_size(low_resolution_image, model_name):
    input_pixel_count = low_resolution_image.width * low_resolution_image.height

    if input_pixel_count <= SR_MODEL_MAX_INPUT_PIXELS:
        return

    max_square_side = int(SR_MODEL_MAX_INPUT_PIXELS ** 0.5)

    raise HTTPException(
        status_code=413,
        detail=(
            f"{model_name} 입력 이미지가 너무 큽니다. "
            f"{max_square_side}x{max_square_side} 이하 또는 {SR_MODEL_MAX_INPUT_PIXELS} pixels 이하 이미지로 테스트해 주세요."))


def run_srgan_inference(low_resolution_image):
    input_tensor = convert_image_to_tensor(low_resolution_image).to(DEVICE)

    with torch.inference_mode():
        output_tensor = SRGAN_GENERATOR(input_tensor)

    output_tensor = normalize_srgan_output(output_tensor)

    return convert_tensor_to_image(output_tensor)


def run_realesrgan_inference(low_resolution_image):
    input_tensor = convert_rgb_image_to_bgr_tensor(low_resolution_image).to(DEVICE)

    with torch.inference_mode():
        output_tensor = REALESRGAN_GENERATOR(input_tensor)

    output_tensor = output_tensor.detach().float().cpu().clamp(0.0, 1.0)

    return convert_bgr_tensor_to_rgb_image(output_tensor)


def encode_png(image):
    output_stream = io.BytesIO()
    image.save(output_stream, format="PNG")

    return output_stream.getvalue()


def convert_image_to_tensor(image):
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)

    return image_tensor


def convert_rgb_image_to_bgr_tensor(image):
    image_array = np.asarray(image, dtype=np.float32)[:, :, ::-1] / 255.0
    image_array = np.ascontiguousarray(image_array.transpose(2, 0, 1))

    return torch.from_numpy(image_array).unsqueeze(0)


def normalize_srgan_output(output_tensor):
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
