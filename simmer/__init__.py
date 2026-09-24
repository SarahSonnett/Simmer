"""Simmer -- simulate a sky survey's detections of a synthetic asteroid population.

Simmer is the *second* pipeline downstream of SynthPop: it ingests a
``<name>_synthpop.csv`` catalog of synthetic Main-Belt asteroids and "replays"
a real survey against it -- recreating the survey's cadence, observing geometry,
infrared flux model, rotational lightcurve, and sensitivity -- to determine how
many synthetic bodies the survey would actually have detected.

The first survey backend is the fully-cryogenic WISE/NEOWISE survey (2010),
which carries the 12 um (W3) and 22 um (W4) thermal bands the NEATM flux model
is built around. The :mod:`simmer.surveys` subpackage is pluggable so other
surveys (Pan-STARRS, ATLAS) can be added behind the same engine.

Stages (see :mod:`simmer.pipeline`):

1. **Field of view** -- which objects fell in which frames, from cached survey
   frame pointings/timestamps, prefiltered to |ecliptic latitude| < 30 deg
   (:mod:`simmer.geometry`, :mod:`simmer.ephemeris`).
2. **Thermal flux** -- NEATM (Harris 1998) in-band flux from the assigned
   physical parameters and the observing geometry (:mod:`simmer.thermal`).
3. **Lightcurve** -- modulate the flux by a rotational lightcurve whose
   amplitude follows Sheppard & Jewitt (2004) (:mod:`simmer.lightcurve`).
4. **Detection** -- cut below the per-band 5-sigma sensitivity limit and drop
   ~1% to approximate bad-pixel photometry corruption (:mod:`simmer.detect`).
"""

from .config import SimConfig
from .pipeline import run_simmer

__all__ = ["SimConfig", "run_simmer"]
__version__ = "0.1.0"
