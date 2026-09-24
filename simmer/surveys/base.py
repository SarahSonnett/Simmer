"""Survey backend interface + registry.

A survey backend supplies the three survey-specific ingredients the otherwise
generic engine needs:

* **frame metadata** -- pointings (ra/dec), timestamps (mjd) and band per frame,
  prefiltered to the near-ecliptic region;
* **bands** -- which filters this survey/phase actually has;
* **sensitivity** -- the per-band single-frame 5-sigma limit (Jy).

Register a backend with ``@register("key")`` and select it via
``SimConfig.survey``. This is the seam that makes Simmer adaptable to
Pan-STARRS, ATLAS, etc. without touching the engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd

_REGISTRY: dict[str, type["Survey"]] = {}


def register(key: str):
    def deco(cls):
        _REGISTRY[key] = cls
        cls.key = key
        return cls
    return deco


def get_survey(key: str) -> "Survey":
    if key not in _REGISTRY:
        raise KeyError(f"unknown survey {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


class Survey(ABC):
    """Abstract survey backend."""

    key: str = ""
    default_bands: tuple = ()
    #: per-band 5-sigma single-frame limit, Jy
    default_sensitivity_jy: dict = {}

    @abstractmethod
    def load_frames(self, cfg) -> pd.DataFrame:
        """Return frame metadata: columns mjd, ra (deg), dec (deg), band.

        Implementations should honour ``cfg.frames_cache`` (read/write a local
        cache) and ``cfg.max_ecl_lat_deg`` (near-ecliptic prefilter), and only
        hit the network when ``cfg.fetch_frames`` is True.
        """

    def bands(self, cfg) -> list:
        return list(cfg.bands) if cfg.bands else list(self.default_bands)

    def sensitivity(self, cfg) -> dict:
        return dict(cfg.sensitivity_jy) if cfg.sensitivity_jy else dict(self.default_sensitivity_jy)
