"""Tests for the Image Classification API.

The ResNet-50 model is mocked so that the test suite runs quickly and without
requiring GPU resources or network access to download pre-trained weights.
"""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_classifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_PREDICTIONS = [
    {"class_name": "tabby cat", "confidence": 0.85, "class_index": 281},
    {"class_name": "tiger cat", "confidence": 0.10, "class_index": 282},
    {"class_name": "Egyptian cat", "confidence": 0.03, "class_index": 285},
    {"class_name": "lynx", "confidence": 0.01, "class_index": 291},
    {"class_name": "Persian cat", "confidence": 0.01, "class_index": 283},
]


def _make_mock_classifier() -> MagicMock:
    mock = MagicMock()
    mock.predict.return_value = _MOCK_PREDICTIONS
    return mock


def _jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Create a small JPEG image as bytes (random pixels)."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Return a TestClient with the classifier dependency overridden."""
    mock_clf = _make_mock_classifier()
    app.dependency_overrides[get_classifier] = lambda: mock_clf
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_body(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestPredict:
    def test_valid_image_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("cat.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("cat.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        data = response.json()
        assert "filename" in data
        assert "content_type" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

    def test_predictions_have_required_fields(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("cat.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        pred = response.json()["predictions"][0]
        assert "class_name" in pred
        assert "confidence" in pred
        assert "class_index" in pred

    def test_filename_echoed(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("my_cat.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.json()["filename"] == "my_cat.jpg"

    def test_top_k_parameter(self, client: TestClient) -> None:
        mock_clf = _make_mock_classifier()
        mock_clf.predict.return_value = _MOCK_PREDICTIONS[:3]
        app.dependency_overrides[get_classifier] = lambda: mock_clf

        response = client.post(
            "/predict",
            files={"file": ("cat.jpg", _jpeg_bytes(), "image/jpeg")},
            params={"top_k": 3},
        )
        assert response.status_code == 200
        # Verify top_k was forwarded to the model
        mock_clf.predict.assert_called_once()
        _, kwargs = mock_clf.predict.call_args
        assert kwargs.get("top_k") == 3

    def test_non_image_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("document.pdf", b"%PDF-1.4 content", "application/pdf")},
        )
        assert response.status_code == 400

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_model_error_returns_422(self, client: TestClient) -> None:
        mock_clf = MagicMock()
        mock_clf.predict.side_effect = ValueError("corrupt image")
        app.dependency_overrides[get_classifier] = lambda: mock_clf

        response = client.post(
            "/predict",
            files={"file": ("bad.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 422
