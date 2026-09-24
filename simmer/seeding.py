"""Deterministic, batch-invariant randomness keyed by object/frame identity.

For results to be reproducible *and* independent of how a run is split across
workers (Step 4 parallelism), randomness must depend on *what* is being drawn
for (an object's global index, a frame's global index) rather than on the order
in which a single RNG stream is consumed. We get that from a vectorized
splitmix64 hash: ``hash01(seed, stream, key_a, key_b, ...)`` maps integer keys to
a uniform in [0, 1), identically no matter how the arrays are batched.

This is a hashing PRNG, adequate for assigning rotation phases and bad-pixel
draws; it is not a substitute for a full Generator where stream quality matters.
"""

from __future__ import annotations

import numpy as np

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)

# Distinct stream ids so different quantities drawn for the same object/frame
# are uncorrelated. Add new ones here rather than reusing.
STREAM_ROT_PERIOD = np.uint64(1)
STREAM_ROT_PHASE = np.uint64(2)
STREAM_BAD_PIXEL = np.uint64(3)
STREAM_PHOT_NOISE = np.uint64(4)


def _mix(z: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer on a uint64 array (arithmetic wraps mod 2**64)."""
    z = (z ^ (z >> np.uint64(30))) * _C1
    z = (z ^ (z >> np.uint64(27))) * _C2
    return z ^ (z >> np.uint64(31))


def hash01(seed: int, stream: np.uint64, *keys: np.ndarray) -> np.ndarray:
    """Uniform draws in [0, 1) from integer keys, deterministic and vectorized.

    ``seed`` and ``stream`` fix the substream; ``keys`` are broadcast integer
    arrays (e.g. object index, frame index) identifying *what* the draw is for.
    """
    with np.errstate(over="ignore"):
        h = (np.uint64(np.uint64(seed) & _MASK) ^ (stream * _GOLDEN))
        h = _mix(h)
        for k in keys:
            k = np.asarray(k).astype(np.uint64)
            h = _mix(h ^ (k + _GOLDEN))
    # top 53 bits -> double in [0, 1)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
