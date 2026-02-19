from enum import Enum

__all__ = [
    "PsMethod",
    "ShpMethod",
    "UnwrapMethod",
]


class PsMethod(str, Enum):
    """Method for selecting persistent scatterers."""

    AMPLITUDE_DISPERSION = "amplitude_dispersion"
    SCR = "scr"


class ShpMethod(str, Enum):
    """Method for finding SHPs during phase linking."""

    GLRT = "glrt"
    KS = "ks"
    RECT = "rect"
    GAUSSIAN = "gaussian"
    # Alias for no SHP search
    NONE = "rect"


class UnwrapMethod(str, Enum):
    """Phase unwrapping method."""

    SNAPHU = "snaphu"
    ICU = "icu"
    PHASS = "phass"
    SPURT = "spurt"
    WHIRLWIND = "whirlwind"
    SPURS = "spurs"
