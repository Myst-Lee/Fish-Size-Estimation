from __future__ import annotations
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from src.prediction import available_species, load_artifacts, predict_length_cm
from src.segmentation import (
    generate_sam_masks,
    largest_sam_mask_to_segmentation,
    load_sam_mask_generator,
    overlay_mask,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
SAM_CHECKPOINT = APP_DIR / "sam" / "sam_vit_h_4b8939.pth"


st.set_page_config(page_title="Fish Length Predictor", layout="wide")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def get_sam_generator():
    try:
        checkpoint_url = st.secrets.get("SAM_CHECKPOINT_URL", None)
    except Exception:
        checkpoint_url = None
    return load_sam_mask_generator(SAM_CHECKPOINT, checkpoint_url=checkpoint_url)


def read_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


st.title("Fish Length Predictor")

with st.sidebar:
    st.header("Model")
    st.caption("Uses the final RandomForest model trained with morphology, species, and ratio features.")
    st.caption("Segmentation is generated automatically with SAM.")

try:
    artifacts = get_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

species_options = available_species(artifacts.species_columns)
if not species_options:
    st.error("No species columns were found in final_species_columns.json.")
    st.stop()

left, right = st.columns([0.8, 1.2])

with left:
    uploaded_image = st.file_uploader("Fish image", type=["jpg", "jpeg", "png"])
    species = st.selectbox("Species", species_options)

if uploaded_image is None:
    st.info("Upload a fish image to start.")
    st.stop()

image_rgb = read_image(uploaded_image)
image_height, image_width = image_rgb.shape[:2]

with right:
    st.image(image_rgb, caption="Uploaded image", use_container_width=True)

mask = None
segmentation = None
bbox = None

with st.spinner("Generating fish mask with SAM..."):
    try:
        sam_generator = get_sam_generator()
        masks = generate_sam_masks(image_rgb, sam_generator)
        result = largest_sam_mask_to_segmentation(masks)
    except Exception as error:  # Streamlit should surface deployment setup issues cleanly.
        st.error(f"SAM segmentation failed: {error}")
        st.stop()

if result is None:
    st.error("SAM did not produce a usable fish mask.")
    st.stop()
segmentation, bbox, mask = result

preview, metrics = st.columns([1.1, 0.9])
with preview:
    st.image(overlay_mask(image_rgb, mask), caption="Selected mask overlay", use_container_width=True)

with metrics:
    try:
        prediction, feature_frame, morphology, ratios = predict_length_cm(
            species=species,
            segmentation=segmentation,
            bbox=bbox,
            image_width=image_width,
            image_height=image_height,
            artifacts=artifacts,
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.metric("Predicted length", f"{prediction:.2f} cm")
    st.caption("Prediction quality depends heavily on the selected mask and consistent image scale.")

with st.expander("Extracted morphology and ratio features"):
    st.dataframe(
        {
            "Feature": list(morphology.keys()) + list(ratios.keys()),
            "Value": list(morphology.values()) + list(ratios.values()),
        },
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Model input row"):
    st.dataframe(feature_frame, use_container_width=True, hide_index=True)
