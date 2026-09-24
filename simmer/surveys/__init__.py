"""Pluggable survey backends.

Importing this package registers the available backends. Select one via
``SimConfig.survey`` and obtain it with :func:`simmer.surveys.base.get_survey`.
"""

from .base import Survey, register, get_survey
from . import neowise          # noqa: F401  (registers "neowise_cryo")
from . import panstarrs        # noqa: F401  (registers "panstarrs", stub)
from . import atlas            # noqa: F401  (registers "atlas", stub)

__all__ = ["Survey", "register", "get_survey"]
