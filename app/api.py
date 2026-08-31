import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.model_loader import ModelNotFoundError
from app.predictor import InvalidImageError, predict_image

from app.schemas import HealthResponse


app = FastAPI(
    title="Vision Inspector Agent",
    version="0.1.0",
    description="A FastAPI application for image inspection and analysis.",
)


@app.get("/")
def root():
    return {"message": "Vision Inspector API"}

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")

@app.get("/model/info")
def model_info():
    """返回当前模型信息，不加载模型、不执行检测，仅读取 config 配置。"""
    settings = Settings.from_env()
    model_path = settings.model_path
    return {
        "model": {
            "path": str(model_path),
            "exists": model_path.is_file(),
            "size_bytes": model_path.stat().st_size if model_path.is_file() else None,
        },
        "inference": {
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "image_size": settings.image_size,
            "device": settings.device,
        },
        "output_dir": str(settings.output_dir),
    }

@app.post("/upload")
async def upload(file: UploadFile):
    return {
        "filename": file.filename,
        "content_type": file.content_type,
    }


@app.post("/predictions")
async def create_prediction(file: UploadFile = File(...)):
    """上传单张图片，临时落盘后调用 app.predictor.predict_image() 推理，返回统一 JSON 结果。"""
    settings = Settings.from_env()
    suffix = Path(file.filename or "").suffix.lower()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = await run_in_threadpool(predict_image, tmp_path, settings=settings)
        return result.to_dict()
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
