from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .feature_extraction import extract_features, ratio_features_from_morphology


MORPHOLOGY_FEATURES = ["pca_length", "pca_height", "area", "perimeter", "solidity"]
RATIO_FEATURES = [
    "aspect_ratio",
    "relative_depth",
    "area_ratio",
    "perimeter_ratio",
    "compactness",
    "shape_factor",
]


@dataclass(frozen=True)
class FishLengthArtifacts:
    model: object
    scaler: object
    species_columns: list[str]
    feature_columns: list[str]


def load_artifacts(model_dir: str | Path) -> FishLengthArtifacts:
    model_path = Path(model_dir)
    with (model_path / "final_species_columns.json").open("r", encoding="utf-8") as file:
        species_columns = json.load(file)
    with (model_path / "final_feature_columns.json").open("r", encoding="utf-8") as file:
        feature_columns = json.load(file)

    return FishLengthArtifacts(
        model=joblib.load(model_path / "final_random_forest_model.joblib"),
        scaler=joblib.load(model_path / "final_scaler.joblib"),
        species_columns=species_columns,
        feature_columns=feature_columns,
    )


def available_species(species_columns: list[str]) -> list[str]:
    return sorted(column.removeprefix("species_") for column in species_columns)


def build_feature_frame(
    species: str,
    segmentation: list[int] | list[float],
    bbox: list[float],
    image_width: int,
    image_height: int,
    artifacts: FishLengthArtifacts,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    morphology = extract_features(segmentation, bbox, image_width, image_height)
    if morphology is None:
        raise ValueError("Could not extract morphology features from the selected mask.")

    ratios = ratio_features_from_morphology(morphology)
    sample: dict[str, float | int] = {}

    for feature in MORPHOLOGY_FEATURES:
        sample[feature] = morphology[feature]
    for feature in RATIO_FEATURES:
        sample[feature] = ratios[feature]
    for column in artifacts.species_columns:
        sample[column] = 1 if column == f"species_{species}" else 0

    frame = pd.DataFrame([sample])
    for column in artifacts.feature_columns:
        if column not in frame.columns:
            frame[column] = 0

    return frame[artifacts.feature_columns], morphology, ratios


def predict_length_cm(
    species: str,
    segmentation: list[int] | list[float],
    bbox: list[float],
    image_width: int,
    image_height: int,
    artifacts: FishLengthArtifacts,
) -> tuple[float, pd.DataFrame, dict[str, float], dict[str, float]]:
    feature_frame, morphology, ratios = build_feature_frame(
        species=species,
        segmentation=segmentation,
        bbox=bbox,
        image_width=image_width,
        image_height=image_height,
        artifacts=artifacts,
    )
    scaled = artifacts.scaler.transform(feature_frame)
    prediction = float(artifacts.model.predict(scaled)[0])
    return prediction, feature_frame, morphology, ratios
