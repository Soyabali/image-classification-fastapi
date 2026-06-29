import io
from typing import List

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


class ImageClassifier:
    """Wraps a pre-trained ResNet-50 model for ImageNet classification."""

    def __init__(self) -> None:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        self.categories: List[str] = weights.meta["categories"]
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image_bytes: bytes, top_k: int = 5) -> List[dict]:
        """Return the top-k ImageNet predictions for the supplied image bytes.

        Args:
            image_bytes: Raw bytes of the image to classify.
            top_k: Number of top predictions to return.

        Returns:
            A list of dicts, each containing ``class_name``, ``confidence``,
            and ``class_index``, sorted by descending confidence.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        top_k = min(top_k, len(self.categories))
        top_probs, top_indices = torch.topk(probabilities, top_k)

        return [
            {
                "class_name": self.categories[idx],
                "confidence": round(float(prob), 4),
                "class_index": int(idx),
            }
            for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
        ]
