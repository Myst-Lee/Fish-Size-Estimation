from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def mask_to_segmentation(mask: np.ndarray) -> tuple[list[int], list[float]] | None:
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    segmentation_points = contour.flatten().astype(int).tolist()
    bbox = [float(x), float(y), float(w), float(h)]
    return segmentation_points, bbox


def largest_sam_mask_to_segmentation(masks: list[dict[str, Any]]) -> tuple[list[int], list[float], np.ndarray] | None:
    if not masks:
        return None

    largest_mask = max(masks, key=lambda item: item["area"])
    segmentation_mask = largest_mask["segmentation"].astype(np.uint8)
    converted = mask_to_segmentation(segmentation_mask)
    if converted is None:
        return None

    segmentation_points, bbox = converted
    return segmentation_points, bbox, segmentation_mask


def load_sam_mask_generator(checkpoint_path: str | Path, model_type: str = "vit_h") -> Any:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {checkpoint}. Download sam_vit_h_4b8939.pth "
            "and place it in the sam/ folder, or use manual mask upload."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)


def generate_sam_masks(image_rgb: np.ndarray, mask_generator: Any) -> list[dict[str, Any]]:
    return mask_generator.generate(image_rgb)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (36, 160, 237)) -> np.ndarray:
    overlay = image_rgb.copy()
    colored = np.zeros_like(image_rgb)
    colored[:, :] = np.array(color, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (0.55 * image_rgb[mask_bool] + 0.45 * colored[mask_bool]).astype(np.uint8)
    return overlay
