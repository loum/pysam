"""Facebook Research context-based model lookup."""

from dataclasses import dataclass
from enum import Enum

import py_sam.model.hiera


@dataclass(frozen=True)
class FbrSamEnum(str, Enum):
    """Facebook Research model enumerations custom to the model type."""

    HIERA_B = "hiera_b"
    HIERA_L = "hiera_l"
    HIERA_S = "hiera_s"
    HIERA_T = "hiera_t"


class FbrSam(Enum):
    """Facebook Research model enumerations."""

    HIERA_B = py_sam.model.hiera.HieraBasePlus
    HIERA_L = py_sam.model.hiera.HieraLarge
    HIERA_S = py_sam.model.hiera.HieraSmall
    HIERA_T = py_sam.model.hiera.HieraTiny
