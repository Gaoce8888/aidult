"""Traditional image forensics algorithms.

This module implements lightweight, heuristic-based checks that quickly flag
obvious manipulations before expensive deep models are invoked.
"""
from typing import Dict, List

import cv2
import numpy as np
from loguru import logger


class TraditionalDetector:  # pylint: disable=too-few-public-methods
    """Rule-based and statistical detection methods."""

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Run all traditional detectors and return aggregated risk items.

        Args:
            image: BGR image as numpy array.

        Returns:
            A list of dict objects describing risk factors.
        """
        risks: List[Dict] = []

        try:
            risks.extend(self._detect_compression_artifacts(image))
            risks.extend(self._detect_noise_patterns(image))
            risks.extend(self._detect_edge_inconsistency(image))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Traditional detection failed: {}", exc)

        return risks

    # ---------------------------------------------------------------------
    # Internal methods (stub implementations)
    # ---------------------------------------------------------------------

    def _detect_compression_artifacts(self, image: np.ndarray) -> List[Dict]:
        """Detect JPEG double compression artifacts (stub)."""
        # TODO: Implement actual algorithm
        _ = image  # suppress unused warning
        return []

    def _detect_noise_patterns(self, image: np.ndarray) -> List[Dict]:
        """Detect abnormal noise patterns (stub)."""
        _ = image
        return []

    def _detect_edge_inconsistency(self, image: np.ndarray) -> List[Dict]:
        """Detect edge inconsistencies (stub)."""
        _ = image
        return []