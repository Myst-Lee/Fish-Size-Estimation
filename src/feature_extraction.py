from __future__ import annotations

import ast
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, skeletonize
from sklearn.decomposition import PCA


def fix_segmentation(segmentation: Any) -> np.ndarray | None:
    """Convert COCO-style segmentation points to an Nx2 contour array."""
    try:
        if isinstance(segmentation, str):
            segmentation = ast.literal_eval(segmentation)
        while isinstance(segmentation, list) and len(segmentation) == 1:
            segmentation = segmentation[0]
        contour = np.array(segmentation, dtype=np.float32).flatten()
        if len(contour) % 2 != 0:
            return None
        return contour.reshape(-1, 2)
    except (SyntaxError, ValueError, TypeError):
        return None


def contour_to_local_mask(contour: np.ndarray, padding: int = 10) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    w_pad = w + 2 * padding
    h_pad = h + 2 * padding

    local_contour = contour.copy()
    local_contour[:, 0] -= x0
    local_contour[:, 1] -= y0

    mask = np.zeros((h_pad, w_pad), dtype=np.uint8)
    cv2.fillPoly(mask, [local_contour.astype(np.int32)], 255)
    return mask


def mask_to_skeleton(mask: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(mask > 0)
    return skeleton.astype(np.uint8)


def compute_skeleton_length(skeleton: np.ndarray) -> float:
    return float(np.count_nonzero(skeleton))


def compute_medial_axis_length(mask: np.ndarray) -> tuple[float, np.ndarray]:
    medial, _ = medial_axis(mask > 0, return_distance=True)
    return float(np.count_nonzero(medial)), medial.astype(np.uint8)


def compute_pca_length_height(contour: np.ndarray) -> tuple[float, float]:
    pts = contour.astype(np.float32)
    pca = PCA(n_components=2)
    pca.fit(pts)

    principal_axis_1 = pca.components_[0]
    projections_1 = pts @ principal_axis_1
    pca_length = projections_1.max() - projections_1.min()

    principal_axis_2 = pca.components_[1]
    projections_2 = pts @ principal_axis_2
    pca_height = projections_2.max() - projections_2.min()

    return float(pca_length), float(pca_height)


def compute_body_width_profile(
    mask: np.ndarray, centerline: np.ndarray, sample_step: int = 100
) -> tuple[list[float], float, float]:
    dist_map = distance_transform_edt(mask)
    ys, xs = np.where(centerline > 0)
    xs = xs[::sample_step]
    ys = ys[::sample_step]

    widths = [float(dist_map[y, x] * 2) for x, y in zip(xs, ys)]
    if not widths:
        return [], 0.0, 0.0
    return widths, float(np.mean(widths)), float(np.max(widths))


def extract_features(
    segmentation: Any, bbox: list[float] | None, image_width: int, image_height: int
) -> dict[str, float] | None:
    # bbox, image_width, and image_height are kept for parity with the notebook API.
    _ = bbox, image_width, image_height

    contour = fix_segmentation(segmentation)
    if contour is None or len(contour) < 3:
        return None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    mask = contour_to_local_mask(contour)
    skeleton = mask_to_skeleton(mask)
    skeleton_length = compute_skeleton_length(skeleton)
    _, _, skeleton_max_width = compute_body_width_profile(mask, skeleton)

    medial_axis_length, medial_axis_img = compute_medial_axis_length(mask)
    _, _, medial_max_width = compute_body_width_profile(mask, medial_axis_img)

    pca_length, pca_height = compute_pca_length_height(contour)

    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "hull_area": float(hull_area),
        "solidity": float(solidity),
        "skeleton_length": float(skeleton_length),
        "skeleton_height": float(skeleton_max_width),
        "medial_axis_length": float(medial_axis_length),
        "medial_axis_height": float(medial_max_width),
        "pca_length": float(pca_length),
        "pca_height": float(pca_height),
    }


def ratio_features_from_morphology(features: dict[str, float]) -> dict[str, float]:
    eps = 1e-6
    pca_length = features["pca_length"]
    pca_height = features["pca_height"]
    area = features["area"]
    perimeter = features["perimeter"]

    return {
        "aspect_ratio": pca_length / (pca_height + eps),
        "relative_depth": pca_height / (pca_length + eps),
        "area_ratio": area / ((pca_length**2) + eps),
        "perimeter_ratio": perimeter / (pca_length + eps),
        "compactness": (perimeter**2) / (area + eps),
        "shape_factor": 4 * np.pi * area / ((perimeter**2) + eps),
    }
