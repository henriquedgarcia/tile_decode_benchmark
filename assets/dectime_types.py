from enum import Enum
from typing import Any, Union, NamedTuple

# todo: mover enums para dectime.py? Acho que as enums não.


class Check(Enum):
    ORIGINAL = 0
    LOSSLESS = 1
    COMPRESSED = 2
    SEGMENT = 3
    DECTIME = 4


