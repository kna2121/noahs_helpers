from dataclasses import dataclass
from enum import Enum


class Kind(Enum):
    Helper = "H"
    Noah = "N"


@dataclass(frozen=True)
class PlayerView:
    id: int
    kind: Kind
