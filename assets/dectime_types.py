from enum import Enum
from typing import Any, Union

# todo: mover enums para dectime.py? Acho que as enums não.


class Check(Enum):
    ORIGINAL = 0
    LOSSLESS = 1
    COMPRESSED = 2
    SEGMENT = 3
    DECTIME = 4


class ErrorMetric(Enum):
    RMSE = 0
    NRMSE = 1
    SSE = 2


class Dataframes(Enum):
    STATS_DATAFRAME = 'df_stats'
    FITTED_DATAFRAME = 'df_dist'
    PAPER_DATAFRAME = 'df_paper'
    DATA_DATAFRAME = 'df_data'


class DectimeFactors:
    # Usado no Dectime Analysis
    _config = None
    _name = None
    rate_control = None

    video_name = Union[str, None]
    pattern = Union[str, None]
    quality = Union[int, None]
    tile = Union[int, None]
    chunk = Union[int, None]

    def __init__(self, rate_control):
        self.rate_control = rate_control

    def clear(self):
        self.video_name = None
        self.pattern = None
        self.quality = None
        self.tile = None
        self.chunk = None

    def name(self, base_name: Union[str, None] = None,
             ext: Union[str, None] = None,
             other: Any = None,
             separator='_'):
        name = f'{base_name}' if base_name else None
        if self.video_name:
            name = (f'{name}{separator}{self.video_name}'
                    if name else f'{self.video_name}')
        if self.pattern:
            name = (f'{name}{separator}{self.pattern}'
                    if name else f'{self.pattern}')
        if self.quality:
            name = (f'{name}{separator}{self.rate_control}{self.quality}'
                    if name else f'{self.rate_control}{self.quality}')
        if self.tile:
            name = (f'{name}{separator}tile{self.tile}'
                    if name else f'tile{self.tile}')
        if self.chunk:
            name = (f'{name}{separator}chunk{self.chunk}'
                    if name else f'chunk{self.chunk}')
        if other:
            name = (f'{name}{separator}{other}'
                    if name else f'{other}')
        if ext:
            name = (f'{name}.{ext}'
                    if name else f'{ext}')

        self._name = name
        return name
