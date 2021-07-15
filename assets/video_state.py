from itertools import product as prod
from pathlib import Path
from typing import Any, Dict, List, Union
from abc import ABC, abstractmethod
from assets.util import splitx, AbstractConfig


class Frame:
    _scale: str
    w: int
    h: int

    def __init__(self, scale: Union[str, tuple]):
        """

        :param scale: a string like "1200x600"
        """
        if isinstance(scale, str):
            self.scale = scale
        elif isinstance(scale, tuple):
            self.scale = f'{scale[0]}x{scale[1]}'

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale: str):
        self._scale = scale
        self.w, self.h = splitx(scale)


class Tile:
    def __init__(self, idx: int, scale: Frame, pos: tuple):
        self.idx = idx
        self.scale = scale
        self.x: float = pos[0]
        self.y: float = pos[1]
        self.w: int = scale.w
        self.h: int = scale.h


class Tiling:
    _pattern_list: List[str]
    _pattern: str
    frame: Frame
    tile_scale: Frame
    total_tiles: int
    tiles_list: List[Tile]
    m: int
    n: int

    def __init__(self, pattern, frame: Frame):
        self.pattern = pattern
        self.frame = frame

    @property
    def tiles_list(self):
        self.tile_scale = Frame((self.frame.w // self.m,
                                 self.frame.h // self.n))
        pos_iterator = prod(range(0, self.frame.h, self.tile_scale.h),
                            range(0, self.frame.w, self.tile_scale.w))
        tiles_list = [Tile(idx, self.tile_scale, (x, y))
                      for idx, (y, x) in enumerate(pos_iterator)]
        return tiles_list

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self._pattern = pattern
        self.m, self.n = splitx(pattern)
        self.total_tiles = self.m * self.n


class Video:
    _name: str
    group: int
    original: str
    offset: int
    duration: int
    chunks: range
    tiling: Tile
    _video_info: Dict[str, Any]

    def __init__(self, name, video_info: Dict[str, Any]):
        self.video_info = video_info
        self.name = name

    @property
    def video_info(self):
        return self._video_info

    @video_info.setter
    def video_info(self, video_info: Dict[str, Any]):
        self.group = video_info['group']
        self.original = video_info['original']
        self.offset = video_info['offset']
        self.duration = video_info['duration']
        self.chunks = range(1, (self.duration + 1))

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name


class Factors:
    video: Video = None
    tiling: Tiling = None
    quality: int = None
    tile: Tile = None
    chunk: int = None
    rate_control: str = None

    @property
    def name(self) -> str:
        return self.video.name

    @name.setter
    def name(self, name: str):
        self.video.name = name

    @property
    def pattern(self) -> str:
        return self.tiling.pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self.tiling.pattern = pattern

    @property
    def tile_id(self) -> int:
        return self.tile.idx

    @property
    def state(self):
        state = None
        if self.name:
            state = (f'{state}_{self.name}'
                     if state else f'{self.name}')
        if self.pattern:
            state = (f'{state}_{self.pattern}'
                     if state else f'{self.pattern}')
        if self.quality:
            state = (f'{state}_{self.rate_control}{self.quality}'
                     if state else f'{self.rate_control}{self.quality}')
        if self.tile:
            state = (f'{state}_tile{self.tile_id}'
                     if state else f'tile{self.tile_id}')
        if self.chunk:
            state = (f'{state}_chunk{self.chunk}'
                     if state else f'chunk{self.chunk}')
        return state

    def get_factors(self):
        name = self.name
        pattern = self.pattern
        quality = self.quality
        tile = self.tile_id
        chunk = self.chunk
        return name, pattern, quality, tile, chunk

    def get_name(self, base_name: Union[str, None] = None,
                 ext: Union[str, None] = None,
                 other: Any = None,
                 separator='_'):
        name = self.state.replace('_', separator)
        if base_name:
            name = f'{base_name}{separator}{name}'
        if other:
            name = f'{name}{separator}{other}'
        if ext:
            name = f'{name}.{ext}'
        return name


class Params:
    """
    Interface
    ---------
    This interface represent the constants values for all simulation.
    """
    scale: str
    project: Path
    projection: str
    fps: int
    gop: int
    rate_control: str
    videos_dict: dict


class Paths(Params, Factors):
    """
    original_folder: Folder that contain the original files.
    lossless_folder: Folder to put intermediate lossless full-frame video.
    compressed_folder: Folder to put compressed tiles_list.
    segment_folder: Folder to put the segments of tiles_list.
    dectime_folder: Folder to put decode log.
    """
    _original_folder: Path
    _lossless_folder: Path
    _compressed_folder: Path
    _segment_folder: Path
    _dectime_folder: Path
    _siti_folder: Path

    @property
    def basename(self):
        return Path(f'{self.video.name}_'
                    f'{self.scale}_'
                    f'{self.fps}_'
                    f'{self.pattern}_'
                    f'{self.rate_control}{self.quality}')

    @property
    def original_folder(self) -> Path:
        return self._original_folder

    @property
    def lossless_folder(self) -> Path:
        folder = self.project / self._lossless_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def compressed_folder(self) -> Path:
        folder = self.project / self._compressed_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def segment_folder(self) -> Path:
        folder = self.project / self._segment_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def dectime_folder(self) -> Path:
        folder = self.project / self._dectime_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def siti_folder(self) -> Path:
        folder = self.project / self._siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def original_file(self) -> Path:
        return self.original_folder / self.video.original

    @property
    def lossless_file(self) -> Path:
        return self.lossless_folder / f'{self.name}_{self.scale}_{self.fps}.mp4'

    @property
    def lossless_log(self) -> Path:
        return self.lossless_file.with_suffix('.log')

    @property
    def compressed_file(self) -> Path:
        return self.compressed_folder / f'tile{self.tile_id}.mp4'

    @property
    def compressed_log(self) -> Path:
        return self.compressed_file.with_suffix('.log')

    @property
    def segment_file(self) -> Path:
        return self.segment_folder / f'tile{self.tile_id}_{self.chunk:03}.mp4'

    @property
    def segment_log(self) -> Path:
        return self.segment_folder / f'tile{self.tile_id}.log'

    @property
    def dectime_log(self) -> Path:
        return self.dectime_folder / f'tile{self.tile_id}_{self.chunk:03}.log'

    @property
    def dectime_raw_json(self) -> Path:
        return self.project / 'dectime_raw.json'

    @property
    def siti_results(self) -> Path:
        return self.siti_folder / f'{self.name}_siti_results.csv'

    @property
    def siti_movie(self) -> Path:
        return self.siti_folder / f'{self.name}_siti_movie.mp4'

    @property
    def siti_stats(self) -> Path:
        return self.siti_folder / f'{self.name}_siti_stats.json'


class DectimeLists(Params, Factors):
    _videos_list: List[Video]
    _names_list: List[str]
    _pattern_list: List[str]
    _tiling_list: List[Tiling]
    _quality_list: List[int]
    _tile_list: List[int]
    _chunk_list: List[int]
    videos_dict: dict

    @property
    def videos_list(self):
        return self._videos_list

    @videos_list.setter
    def videos_list(self, videos_dict: Dict[str, dict]):
        self._names_list = [name for name in videos_dict]
        self._videos_list = [Video(name, videos_dict[name])
                             for name in videos_dict]

    @property
    def names_list(self):
        return self._names_list

    @names_list.setter
    def names_list(self, videos_dict: Dict[str, dict]):
        self.videos_list = videos_dict

    @property
    def tiling_list(self):
        return self._tiling_list

    @tiling_list.setter
    def tiling_list(self, pattern_list: List[str]):
        self._pattern_list = pattern_list
        self._tiling_list = [Tiling(pattern, Frame(self.scale))
                             for pattern in pattern_list]

    @property
    def pattern_list(self):
        return self._pattern_list

    @pattern_list.setter
    def pattern_list(self, pattern_list: List[str]):
        self.tiling_list = pattern_list

    @property
    def quality_list(self):
        return self._quality_list

    @quality_list.setter
    def quality_list(self, quality_list: List[int]):
        self._quality_list = quality_list

    @property
    def tiles_list(self):
        return self.tiling.tiles_list

    @property
    def chunk_list(self):
        return self.video.chunks


class AbstractVideoState(ABC, Paths, DectimeLists):
    @abstractmethod
    def __init__(self, config: AbstractConfig):
        pass
