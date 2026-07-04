"""Repo-relative output paths for SVG / figure generators."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS_SVG = REPO / "scripts" / "svg"
DIAGRAMS = REPO / "diagrams"
DSA_DIAGRAMS = REPO / "docs" / "dsa" / "diagrams"
ENGRAM_DIAGRAMS = (
    REPO / "docs" / "material" / "papers" / "engram" / "diagrams"
)
ESS = REPO / "docs" / "figures" / "ess"
FIGURES = REPO / "docs" / "figures"
