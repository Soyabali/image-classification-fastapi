from functools import lru_cache
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.model import ImageClassifier

app = FastAPI(
    title="Image Classification API",
    description=(
        "A RESTful API for image classification using a pre-trained "
        "ResNet-50 deep learning model (ImageNet, 1 000 classes)."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Dependency – lazily loaded and cached for the lifetime of the process
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_classifier() -> ImageClassifier:
    return ImageClassifier()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    class_name: str
    confidence: float
    class_index: int


class PredictionResponse(BaseModel):
    filename: str
    content_type: str
    predictions: List[Prediction]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
async def health_check() -> dict:
    """Return a simple liveness signal."""
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an image",
)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = Query(default=5, ge=1, le=1000, description="Number of top predictions to return"),
    classifier: ImageClassifier = Depends(get_classifier),
) -> PredictionResponse:
    """Upload an image and receive the top-k classification predictions.

    Supported formats: JPEG, PNG, GIF, BMP, TIFF, WebP.
    """
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Please upload an image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        predictions = classifier.predict(image_bytes, top_k=top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not process image: {exc}",
        ) from exc

    return PredictionResponse(
        filename=file.filename or "",
        content_type=file.content_type,
        predictions=[Prediction(**p) for p in predictions],
    )
