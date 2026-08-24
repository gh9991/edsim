"""Data source adapters. Every one returns the canonical schema.

    from edsim.loaders import load
    df = load("mimic_demo", path="data/mimic-iv-ed-demo/2.2")
    df = load("portal", path="challenge.csv", mapping={...})   # 15 Sept

Add a source here and the simulator, metrics and models pick it up for free.
"""
from __future__ import annotations

import pandas as pd

from edsim.loaders import mimic_demo, portal, synthea

REGISTRY = {
    "mimic_demo": mimic_demo.load,
    "synthea": synthea.load,
    "portal": portal.load,
}


def load(source: str, **kwargs) -> pd.DataFrame:
    if source not in REGISTRY:
        raise KeyError(f"unknown source {source!r}; have {sorted(REGISTRY)}")
    return REGISTRY[source](**kwargs)
