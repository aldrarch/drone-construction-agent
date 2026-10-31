"""
Model factory: selects the model backend from config.MODEL_TYPE.

Adding a new model only requires adding a new branch here and a new
implementation file in this package. The rest of the application never
imports a concrete model directly.
"""

from __future__ import annotations

from typing import Any

import config as app_config

from model_interface import ConstructionModel


def build_model(cfg: Any = None) -> ConstructionModel:
    """Return a model instance according to the configured MODEL_TYPE."""
    model_type = app_config.MODEL_TYPE

    if model_type == "clip":
        from models.clip_model import CLIPModel

        return CLIPModel(model_name=app_config.CLIP_MODEL_NAME)

    raise NotImplementedError(
        f"Unknown MODEL_TYPE='{model_type}'. "
        "Supported values: 'clip'."
    )
