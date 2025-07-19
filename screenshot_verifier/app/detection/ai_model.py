"""Deep-learning based screenshot forgery detector."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger


class AIModel:  # pylint: disable=too-few-public-methods
    """Wrapper around a torch model for inference."""

    _MODEL_WEIGHTS = Path(__file__).parent / "weights" / "model.pt"

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()

    @property
    def version(self) -> str:  # noqa: D401
        """Return model version string."""
        return "v0.1-stub"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Tuple[bool, float, List[Dict]]:  # noqa: D401
        """Return authenticity prediction, confidence and risk factors.

        Currently returns dummy output.
        """
        _ = image  # placeholder
        authentic = True
        confidence = 0.5
        risk_items: List[Dict] = []
        return authentic, confidence, risk_items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:  # noqa: D401
        """Load model weights or build architecture placeholder."""
        # TODO: Replace with actual architecture (e.g., EfficientNet + attention)
        logger.info("Loading model weights from {}", self._MODEL_WEIGHTS)

        # As placeholder, use a trivial linear layer
        model = torch.nn.Identity()
        model.to(self.device)
        return model