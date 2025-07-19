"""Fusion engine merging traditional algorithms with deep model outputs."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from loguru import logger

from screenshot_verifier.app.detection.ai_model import AIModel
from screenshot_verifier.app.detection.traditional import TraditionalDetector


class FusionEngine:  # pylint: disable=too-few-public-methods
    """Decision layer that combines multiple detectors."""

    def __init__(self) -> None:
        self._ai = AIModel()
        self._trad = TraditionalDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self, image: np.ndarray, metadata: Dict | None = None
    ) -> Tuple[bool, float, List[Dict]]:  # noqa: D401
        """Fuse predictions from different detectors.

        Args:
            image: BGR image as numpy array.
            metadata: Optional metadata dict.

        Returns:
            authentic (bool): Final authenticity decision.
            confidence (float): 0-1 confidence score.
            risk_items (List[Dict]): Combined risk factors.
        """
        # Run traditional detectors quickly
        trad_risks = self._trad.detect(image)

        # Run AI model (placeholder)
        ai_authentic, ai_conf, ai_risks = self._ai.predict(image)

        # Simple voting/threshold strategy (placeholder)
        if trad_risks:
            authentic = False
            confidence = max(0.2, 1 - ai_conf)  # crude adjustment
        else:
            authentic = ai_authentic
            confidence = ai_conf

        risk_items: List[Dict] = trad_risks + ai_risks

        logger.debug(
            "Fusion output — authentic: {}, confidence: {:.3f}, risks: {}",
            authentic,
            confidence,
            len(risk_items),
        )
        return authentic, confidence, risk_items

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def model_version(self) -> str:  # noqa: D401
        return self._ai.version