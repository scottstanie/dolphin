from enum import Enum

__all__ = [
    "OutputFormat",
    "ShpMethod",
    "UnwrapMethod",
]


class OutputFormat(str, Enum):
    """Output container format for phase-linking outputs."""

    GEOTIFF = "geotiff"
    GEOZARR = "geozarr"


class ShpMethod(str, Enum):
    """Method for finding SHPs during phase linking."""

    GLRT = "glrt"
    KS = "ks"
    RECT = "rect"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method."""

    SNAPHU = "snaphu"
    ICU = "icu"
    PHASS = "phass"
    SPURT = "spurt"
    WHIRLWIND = "whirlwind"
