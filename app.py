"""
app.py
------
Streamlit web application: Bike Verification System.

Runtime pipeline
----------------
1. User uploads a bike image via the UI.
2. The image is preprocessed with the same transforms used at embedding time.
3. A 2048-dim feature vector is produced by a pretrained ResNet50.
4. That vector is compared against every entry in 'bike_embeddings.npy'
   using cosine similarity.
5. The maximum similarity score is taken.
6. Score > SIMILARITY_THRESHOLD → VALID BIKE, otherwise → INVALID BIKE.

Start the app:
    streamlit run app.py
"""

import os
from typing import List, Optional

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Configuration ─────────────────────────────────────────────────────────────

EMBEDDINGS_FILE = "bike_embeddings.npy"
SIMILARITY_THRESHOLD = 0.5

# Must exactly match the transform used in generate_embeddings.py
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


# ── Model & data loading (cached) ─────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model…")
def load_feature_extractor() -> nn.Module:
    """Load ResNet50 (classification head removed) and cache it for the session."""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval()
    return model


@st.cache_data(show_spinner="Loading embeddings…")
def load_embeddings() -> Optional[np.ndarray]:
    """Load precomputed bike embeddings from disk, or return None if missing."""
    if not os.path.isfile(EMBEDDINGS_FILE):
        return None
    return np.load(EMBEDDINGS_FILE)


# ── Core ML functions ─────────────────────────────────────────────────────────


def embed_image(model: nn.Module, image: Image.Image) -> np.ndarray:
    """Return a 2048-dim feature vector for the given PIL image."""
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor).squeeze(0).numpy()
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns a value in [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def max_cosine_similarity(query: np.ndarray, database: np.ndarray) -> float:
    """
    Return the highest cosine similarity between *query* and any embedding
    in *database* (shape: N × D).
    """
    scores = [cosine_similarity(query, ref) for ref in database]
    return max(scores)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Bike Verification System",
    page_icon="🚲",
    layout="centered",
)

st.title("🚲 Bike Verification System")
st.write(
    "Upload a bike image. The system will compare it against the registered "
    "bike model and tell you whether it is a **VALID** or **INVALID** match."
)
st.divider()

# Load model and database embeddings
model = load_feature_extractor()
db_embeddings = load_embeddings()

if db_embeddings is None:
    st.error(
        f"Embeddings database `{EMBEDDINGS_FILE}` not found.  \n"
        "Run the offline step first:  \n"
        "```\npython generate_embeddings.py\n```"
    )
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose an image to verify",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)
    st.divider()

    # ── Inference ─────────────────────────────────────────────────────────────

    with st.spinner("Analysing image…"):
        query_embedding = embed_image(model, image)
        score = max_cosine_similarity(query_embedding, db_embeddings)

    # ── Results ───────────────────────────────────────────────────────────────

    st.subheader("Verification Result")
    st.metric(
        label="Max Cosine Similarity",
        value=f"{score:.4f}",
        help=f"Threshold: {SIMILARITY_THRESHOLD}. Values above this are considered a match.",
    )

    if score > SIMILARITY_THRESHOLD:
        st.success("## ✅  VALID BIKE", icon="✅")
    else:
        st.error("## ❌  INVALID BIKE", icon="❌")
