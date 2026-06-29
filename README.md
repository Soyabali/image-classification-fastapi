# image-classification-fastapi

A **Computer Vision** application for image classification built with **FastAPI**.  
It exposes a RESTful API that accepts image uploads and returns classification predictions from a pre-trained **ResNet-50** deep learning model (ImageNet – 1 000 classes).

---

## Features

- `POST /predict` – Upload any image (JPEG, PNG, GIF, BMP, TIFF, WebP) and receive ranked class predictions with confidence scores.
- `GET /health` – Liveness check endpoint.
- Configurable `top_k` query parameter (default: 5, max: 1 000).
- Interactive API docs at `/docs` (Swagger UI) and `/redoc`.

---

## Project structure

```
image-classification-fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI application & routes
│   └── model.py       # ResNet-50 wrapper (torchvision)
├── tests/
│   ├── __init__.py
│   └── test_api.py    # Pytest test suite (mocked model)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.  
Open `http://127.0.0.1:8000/docs` to explore the interactive Swagger UI.

### 3. Classify an image

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -F "file=@/path/to/your/image.jpg"
```

Example response:

```json
{
  "filename": "image.jpg",
  "content_type": "image/jpeg",
  "predictions": [
    {"class_name": "tabby cat",    "confidence": 0.8512, "class_index": 281},
    {"class_name": "tiger cat",   "confidence": 0.0934, "class_index": 282},
    {"class_name": "Egyptian cat","confidence": 0.0271, "class_index": 285},
    {"class_name": "lynx",        "confidence": 0.0088, "class_index": 291},
    {"class_name": "Persian cat", "confidence": 0.0063, "class_index": 283}
  ]
}
```

---

## Docker

```bash
docker build -t image-classification-fastapi .
docker run -p 8000:8000 image-classification-fastapi
```

---

## Running tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Model

The application uses **ResNet-50** (`IMAGENET1K_V1` weights) from `torchvision`.  
Model weights (~100 MB) are downloaded automatically on first startup and cached in the default PyTorch cache directory (`~/.cache/torch/hub/checkpoints/`).
