"""
generate_embeddings.py
----------------------
Offline step: scan all images in the 'dataset/' folder, generate a 2048-dim
feature vector for each one using a pretrained ResNet50, and persist the
result as 'bike_embeddings.npy'.

Run once before starting the web app:
    python generate_embeddings.py
"""

import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

DATASET_DIR = "dataset"
OUTPUT_FILE = "bike_embeddings.npy"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard ImageNet normalisation used by all torchvision ResNet weights
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def load_feature_extractor() -> nn.Module:
    """
    Return ResNet50 with its final classification head replaced by an
    identity layer so the model outputs raw 2048-dim feature vectors.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model


def embed_image(model: nn.Module, image_path: str) -> np.ndarray:
    """Generate a single 2048-dim embedding for the image at *image_path*."""
    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)  # shape: (1, 3, 224, 224)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0).numpy()  # shape: (2048,)
    return embedding


def collect_image_paths(directory: str) -> List[str]:
    """Return sorted list of supported image paths inside *directory*."""
    return sorted(
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS
    )


def generate_embeddings() -> None:
    # ── Validate dataset directory ────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Dataset folder '{DATASET_DIR}' not found. "
            "Create it and add your reference bike images before running this script."
        )

    image_paths = collect_image_paths(DATASET_DIR)
    if not image_paths:
        raise ValueError(
            f"No supported images found in '{DATASET_DIR}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Found {len(image_paths)} image(s). Loading ResNet50…")
    model = load_feature_extractor()

    # ── Generate embeddings ───────────────────────────────────────────────────
    embeddings: List[np.ndarray] = []
    for path in image_paths:
        print(f"  Processing: {path}")
        embeddings.append(embed_image(model, path))

    # ── Persist to disk ───────────────────────────────────────────────────────
    embeddings_array = np.array(embeddings)  # shape: (N, 2048)
    np.save(OUTPUT_FILE, embeddings_array)

    print(f"\nDone. Saved {len(embeddings)} embedding(s) → '{OUTPUT_FILE}'")
    print(f"Embedding matrix shape: {embeddings_array.shape}")


if __name__ == "__main__":
    generate_embeddings()
