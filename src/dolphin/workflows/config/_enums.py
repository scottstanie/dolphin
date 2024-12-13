from enum import Enum

__all__ = [
    "CallFunc",
    "ShpMethod",
    "UnwrapMethod",
]


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


class CallFunc(str, Enum):
    """Call function for the timeseries method to find reference point."""

    MIN = "min"
    MAX = "max"


class InitMethod(str, Enum):
    MCF = "mcf"
    MST = "mst"


class CostMode(str, Enum):
    DEFO = "defo"
    SMOOTH = "smooth"


class InversionMethod(str, Enum):
    L1 = "L1"
    L2 = "L2"


class SpurtCostType(str, Enum):
    CONSTANT = "constant"
    DISTANCE = "distance"
    CENTROID = "centroid"
