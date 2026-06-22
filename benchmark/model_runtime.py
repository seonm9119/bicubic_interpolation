import base64
import io
import os
import threading
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
SR_MODEL_IDLE_UNLOAD_SECONDS = float(os.environ.get("SR_MODEL_IDLE_UNLOAD_SECONDS", "60"))
DEVICE = torch.device("cuda" if PREFERRED_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

MODEL_LOCK = threading.RLock()
MODEL_UNLOAD_TIMER = None
ACTIVE_MODEL_REQUESTS = 0
LAST_MODEL_USED_AT = None
SRGAN_GENERATOR = None
REALESRGAN_GENERATOR = None


def create_health_response():
    model_state = get_model_state()
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
        "idleUnloadSeconds": SR_MODEL_IDLE_UNLOAD_SECONDS,
        "modelState": model_state,
    }


def get_model_state():
    with MODEL_LOCK:
        if LAST_MODEL_USED_AT is None:
            last_used_seconds_ago = None
        else:
            last_used_seconds_ago = round(time.monotonic() - LAST_MODEL_USED_AT, 2)

        return {
            "activeRequests": ACTIVE_MODEL_REQUESTS,
            "lastUsedSecondsAgo": last_used_seconds_ago,
            "loaded": {
                "srgan": SRGAN_GENERATOR is not None,
                "realesrgan": REALESRGAN_GENERATOR is not None,
            },
        }


def begin_model_request():
    global ACTIVE_MODEL_REQUESTS

    with MODEL_LOCK:
        ACTIVE_MODEL_REQUESTS += 1
        cancel_model_unload_timer_locked()


def finish_model_request():
    global ACTIVE_MODEL_REQUESTS
    global LAST_MODEL_USED_AT

    with MODEL_LOCK:
        ACTIVE_MODEL_REQUESTS = max(0, ACTIVE_MODEL_REQUESTS - 1)
        LAST_MODEL_USED_AT = time.monotonic()
        schedule_model_unload_locked()


def cancel_model_unload_timer_locked():
    global MODEL_UNLOAD_TIMER

    if MODEL_UNLOAD_TIMER is not None:
        MODEL_UNLOAD_TIMER.cancel()
        MODEL_UNLOAD_TIMER = None


def schedule_model_unload_locked():
    global MODEL_UNLOAD_TIMER

    if DEVICE.type != "cuda":
        return

    if SR_MODEL_IDLE_UNLOAD_SECONDS < 0:
        return

    cancel_model_unload_timer_locked()

    if SR_MODEL_IDLE_UNLOAD_SECONDS == 0:
        release_models(force=True)
        return

    MODEL_UNLOAD_TIMER = threading.Timer(SR_MODEL_IDLE_UNLOAD_SECONDS, release_models)
    MODEL_UNLOAD_TIMER.daemon = True
    MODEL_UNLOAD_TIMER.start()


def release_models(force=False):
    global SRGAN_GENERATOR
    global REALESRGAN_GENERATOR

    with MODEL_LOCK:
        if ACTIVE_MODEL_REQUESTS > 0:
            return False

        if not force and LAST_MODEL_USED_AT is not None:
            idle_seconds = time.monotonic() - LAST_MODEL_USED_AT

            if idle_seconds < SR_MODEL_IDLE_UNLOAD_SECONDS:
                schedule_model_unload_locked()
                return False

        had_loaded_models = SRGAN_GENERATOR is not None or REALESRGAN_GENERATOR is not None
        SRGAN_GENERATOR = None
        REALESRGAN_GENERATOR = None

        if DEVICE.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return had_loaded_models


def get_srgan_generator():
    global SRGAN_GENERATOR

    with MODEL_LOCK:
        if SRGAN_GENERATOR is None:
            SRGAN_GENERATOR = load_srgan_generator(CHECKPOINT_PATH, DEVICE)

        return SRGAN_GENERATOR


def get_realesrgan_generator():
    global REALESRGAN_GENERATOR

    with MODEL_LOCK:
        if REALESRGAN_GENERATOR is None:
            REALESRGAN_GENERATOR = load_realesrgan_generator(REALESRGAN_CHECKPOINT_PATH, DEVICE)

        return REALESRGAN_GENERATOR


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
    begin_model_request()
    try:
        output_image = inference_function(low_resolution_image)
    except torch.OutOfMemoryError as exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(status_code=507, detail=f"{model_name} GPU 메모리가 부족합니다. 더 작은 이미지로 테스트해 주세요.") from exception
    finally:
        finish_model_request()

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
    srgan_generator = get_srgan_generator()

    with torch.inference_mode():
        output_tensor = srgan_generator(input_tensor)

    output_tensor = normalize_srgan_output(output_tensor)

    return convert_tensor_to_image(output_tensor)


def run_realesrgan_inference(low_resolution_image):
    input_tensor = convert_rgb_image_to_bgr_tensor(low_resolution_image).to(DEVICE)
    realesrgan_generator = get_realesrgan_generator()

    with torch.inference_mode():
        output_tensor = realesrgan_generator(input_tensor)

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
