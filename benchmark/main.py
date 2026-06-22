from fastapi import FastAPI, File, UploadFile

from model_runtime import (
    create_health_response,
    get_model_state,
    release_models,
    run_realesrgan_inference,
    run_srgan_inference,
    upscale_with_model,
)


app = FastAPI(title="SR Benchmark Inference API")


@app.get("/api/srgan/health")
def srgan_health():
    return create_health_response()


@app.get("/api/realesrgan/health")
def realesrgan_health():
    return create_health_response()


@app.post("/api/models/release")
def release_loaded_models():
    released = release_models(force=True)

    return {
        "success": True,
        "released": released,
        "modelState": get_model_state(),
    }


@app.post("/api/srgan/upscale")
async def upscale_srgan(image: UploadFile = File(...)):
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
