from torch import Tensor
import numpy as np
from dataclasses import dataclass

@dataclass
class SampleCoords:
    station: str
    warmup_start_date: str
    start_date: str
    end_date: str
    warmup_size: int

@dataclass
class SamplePattern:
    """Class defining MachFlowData return pattern."""
    dfeatures: np.ndarray
    dtargets: np.ndarray
    coords: SampleCoords
    sfeatures: np.ndarray | None = None

@dataclass
class BatchCoords:
    station: list[str]
    warmup_start_date: list[str]
    start_date: list[str]
    end_date: list[str]
    warmup_size: list[int]


@dataclass
class BatchPattern:
    """Class defining MachFlowDataModule dataloader return pattern."""
    dfeatures: Tensor
    dtargets: Tensor
    coords: BatchCoords
    sfeatures: Tensor | None = None


@dataclass
class ReturnPattern:
    """Class defining MachFlowData return pattern."""
    dtargets: Tensor
    coords: BatchCoords
