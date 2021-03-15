from enum import Enum
from typing import Any, NamedTuple, Union

from assets.util import splitx


class Role(Enum):
    PREPARE = 'prepare_videos'
    COMPRESS = 'compress'
    SEGMENT = 'segment'
    DECODE = 'decode'
    RESULTS = 'collect_result'
    SITI = 'calcule_siti'


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


class DectimeData(NamedTuple):
    time: Union[list, float] = []
    rate: Union[list, float] = []


class Frame:
    scale: str
    w: int
    h: int

    def __init__(self, scale):
        self.scale = scale
        self.w, self.h = splitx(scale)


class Tile:
    id: int
    w: int
    h: int
    x: int  # position in frame
    y: int  # position in frame

    def __init__(self, idx=1, w=0, h=0, x=0, y=0):
        self.id = idx
        self.w = w
        self.h = h
        self.scale = f'{w}x{h}'
        self.x = x
        self.y = y


class Tiling:
    pattern: str
    m: int
    n: int
    total: int

    def __init__(self, pattern, frame: Frame):
        self.frame = frame
        self.pattern = pattern
        self.m, self.n = splitx(pattern)
        self.total = self.m * self.n
        self.w = int(round(frame.w / self.m))
        self.h = int(round(frame.h / self.n))
        self.tiles_list = self._tiles_list()

    def _tiles_list(self):
        idx = 0
        tiles_list = []
        for y in range(0, self.frame.h, self.h):
            for x in range(0, self.frame.w, self.w):
                tile = Tile(idx=idx, w=self.w, h=self.h, x=x, y=y)
                tiles_list.append(tile)
                idx += 1

        return tiles_list


class Video:
    name: str
    original: int
    offset: int
    duration: int
    chunks: range
    group: int

    def __init__(self, name: str, video_stats: dict):
        self.name = name
        self.original = video_stats['original']
        self.offset = video_stats['offset']
        self.duration = video_stats['duration']
        self.chunks = range(1, (self.duration + 1))
        self.group = video_stats['group']


class DectimeFactors:
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
