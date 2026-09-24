"""ATLAS backend (stub).

ATLAS is an optical all-sky survey with its own cadence (multiple visits per
night, ~30 s exposures) and o/c filters. Like Pan-STARRS, it reuses the
geometry/cadence/detection engine but needs an exposure-list frame source and a
reflected-light flux model rather than NEATM.
"""

from __future__ import annotations

import pandas as pd

from .base import Survey, register


@register("atlas")
class ATLAS(Survey):
    default_bands = ("o", "c")
    default_sensitivity_jy: dict = {}

    def load_frames(self, cfg) -> pd.DataFrame:
        raise NotImplementedError(
            "ATLAS backend is a stub: implement frame ingestion from the ATLAS "
            "exposure list and an optical (reflected-light) flux model."
        )
