"""
Image loading and validation service.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from config import SUPPORTED_IMAGE_EXTENSIONS


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""


def validate_image_path(file_path: str | Path) -> Path:
    """Check that the path exists and has a supported extension."""
    path = Path(file_path)
    if not path.is_file():
        raise ImageLoadError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        )
    return path


def load_image(file_path: str | Path) -> Image.Image:
    """Load and validate a single image."""
    path = validate_image_path(file_path)
    try:
        image = Image.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"Cannot open image {path}: {exc}") from exc

    if image.format is None:
        raise ImageLoadError(f"File {path} is not a valid image")

    if image.size[0] == 0 or image.size[1] == 0:
        raise ImageLoadError(f"Image {path} is empty")

    return image.convert("RGB")


def image_info(file_path: str | Path) -> dict:
    """Return a small dict with metadata about the image without loading it fully."""
    path = validate_image_path(file_path)
    try:
        with Image.open(path) as img:
            return {
                "filename": path.name,
                "format": img.format,
                "width": img.size[0],
                "height": img.size[1],
                "mode": img.mode,
            }
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError(f"Cannot read image info for {path}: {exc}") from exc
