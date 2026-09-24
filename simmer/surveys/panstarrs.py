"""Pan-STARRS backend (stub).

Pan-STARRS is an optical survey, so a real implementation differs from WISE in
two engine-visible ways: (1) frame metadata/footprints come from the PS1
exposure database rather than IRSA, and (2) the flux model is reflected sunlight
(H-G phase function -> apparent V/grizy magnitudes) rather than NEATM thermal
emission. The geometry/cadence/detection machinery is reused unchanged.

Wire ``load_frames`` to the PS1 exposure list and add an optical-flux path
alongside :mod:`simmer.thermal` when implementing.
"""

from __future__ import annotations

import pandas as pd

from .base import Survey, register


@register("panstarrs")
class PanSTARRS(Survey):
    default_bands = ("g", "r", "i", "z", "y")
    default_sensitivity_jy: dict = {}

    def load_frames(self, cfg) -> pd.DataFrame:
        raise NotImplementedError(
            "Pan-STARRS backend is a stub: implement frame ingestion from the "
            "PS1 exposure database and an optical (reflected-light) flux model."
        )
