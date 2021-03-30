from itertools import product as prod
from os import makedirs
from typing import Any, Dict, List, Union

from assets.config import Config
from assets.util import splitx


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
    frame: Frame
    pattern: str
    tile_scale: Frame
    total_tiles: int
    tiles_list: List[Tile]
    _pattern_list: List[str]

    def __init__(self, pattern, frame: Frame):
        self.pattern = pattern
        self.m, self.n = splitx(pattern)
        self.frame = frame
        self.total_tiles = self.m * self.n

    @property
    def tiles_list(self):
        self.tile_scale = Frame((self.frame.w // self.m,
                                 self.frame.h // self.n))
        pos_iterator = prod(range(0, self.frame.h, self.tile_scale.h),
                            range(0, self.frame.w, self.tile_scale.w))
        tiles_list = [Tile(idx, self.tile_scale, (x, y))
                      for idx, (y, x) in enumerate(pos_iterator)]
        return tiles_list


class Video:
    _name: str
    group: int
    original: int
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
    video: Video
    _name: str
    tiling: Tiling
    _pattern: str
    quality: int
    tile: Tile
    _tile_id: int
    chunk: int

    @property
    def name(self) -> str:
        self._name = self.video.name
        return self._name

    @property
    def pattern(self) -> str:
        self._pattern = self.tiling.pattern
        return self.tiling.pattern

    @property
    def tile_id(self) -> int:
        self._tile_id = self.tile.idx
        return self.tile.idx


class Params:
    """
    Interface
    ---------
    This interface represent the constants values for all simulation.
    """
    scale: str
    project: str
    projection: str
    fps: int
    gop: int
    factor: str
    videos_dict: dict


class Paths(Params, Factors):
    """
    original_folder: Folder that contain the original files.
    lossless_folder: Folder to put intermediate lossless full-frame video.
    compressed_folder: Folder to put compressed tiles_list.
    segment_folder: Folder to put the segments of tiles_list.
    dectime_folder: Folder to put decode log.
    """
    _original_folder: str
    _lossless_folder: str
    _compressed_folder: str
    _segment_folder: str
    _dectime_folder: str

    @property
    def basename(self):
        return (f'{self.video.name}_'
                f'{self.scale}_'
                f'{self.fps}_'
                f'{self.pattern}_'
                f'{self.factor}{self.quality}')

    @property
    def original_folder(self) -> str:
        return self._original_folder

    @property
    def original_file(self) -> str:
        return (f'{self.original_folder}/'
                f'{self.video.original}')

    @property
    def lossless_folder(self) -> str:
        folder = f'{self.project}/{self._lossless_folder}'
        makedirs(folder, exist_ok=True)
        return folder

    @property
    def lossless_file(self) -> str:
        return (f'{self.lossless_folder}/'
                f'{self.name}_{self.scale}_{self.fps}.mp4')

    @property
    def compressed_folder(self) -> str:
        folder = f'{self.project}/{self._compressed_folder}/{self.basename}'
        makedirs(folder, exist_ok=True)
        return folder

    @property
    def compressed_file(self):
        return f'{self.compressed_folder}/tile{self.tile_id}.mp4'

    @property
    def segment_folder(self):
        folder = f'{self.project}/{self._segment_folder}/{self.basename}'
        makedirs(folder, exist_ok=True)
        return folder

    @property
    def segment_file(self):
        return f'{self.segment_folder}/tile{self.tile_id}_{self.chunk:03}.mp4'

    @property
    def dectime_folder(self):
        folder = f'{self.project}/{self._dectime_folder}/{self.basename}'
        makedirs(folder, exist_ok=True)
        return folder

    @property
    def dectime_log(self):
        return f'{self.dectime_folder}/tile{self.tile_id}_{self.chunk:03}.log'

    @property
    def dectime_raw_json(self):
        filename = f'{self.project}/dectime_raw.json'
        return filename


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


class VideoState(Paths, DectimeLists):
    def __init__(self, config: Config):
        """
        Class to create tile files path to process.
        :param config: Config object.
        """
        self.config = config
        self.project = f'results/{config.project}'
        self.scale = config.scale
        self.frame = Frame(config.scale)
        self.fps = config.fps
        self.gop = config.gop
        self.factor = config.rate_control
        self.projection = config.projection
        self.videos_dict = config.videos_list

        self.videos_list = config.videos_list
        self.quality_list = config.quality_list
        self.pattern_list = config.pattern_list

        self._original_folder = config.original_folder
        self._lossless_folder = config.lossless_folder
        self._compressed_folder = config.compressed_folder
        self._segment_folder = config.segment_folder
        self._dectime_folder = config.dectime_folder
