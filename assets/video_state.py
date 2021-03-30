import os
from typing import Union

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
    _video: Union[Video, None] = None
    _tiling = Tiling('1x1', Frame('1x1'))
    _quality = 28
    _tile = _tiling.tiles_list[0]
    _chunk = 1

    @property
    def video(self) -> Video:
        assert self._video is not None, ("The video name variable is not "
                                         "defined.")
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def tiling(self) -> Tiling:
        assert self._tiling is not None, "The tiling variable is not defined."
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value

    @property
    def quality(self) -> int:
        assert self._quality is not None, "The quality variable is not defined."
        return self._quality

    @quality.setter
    def quality(self, value):
        self._quality = value

    @property
    def tile(self) -> Tile:
        assert self._tile is not None, "The tile variable is not defined."
        return self._tile

    @tile.setter
    def tile(self, value):
        self._tile = value

    @property
    def chunk(self) -> int:
        assert self._chunk is not None, "The chunk variable is not defined."
        return self._chunk

    @chunk.setter
    def chunk(self, value):
        self._chunk = value

    def clean(self):
        self.video = None
        self.quality = None
        self.tiling = None
        self.tile = None
        self.chunk = None


class Params:
    project: Union[str, None] = None
    frame: Union[Frame, None] = None
    fps: Union[int, None] = None
    gop: Union[int, None] = None
    factor: Union[str, None] = None


class Paths(Factors, Params):
    _original_folder: Union[str, None] = None
    _lossless_folder: Union[str, None] = None
    _compressed_folder: Union[str, None] = None
    _segment_folder: Union[str, None] = None
    _dectime_folder: Union[str, None] = None

    @property
    def basename(self):
        return (f'{self.video.name}_'
                f'{self.frame.scale}_'
                f'{self.fps}_'
                f'{self.tiling.pattern}_'
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
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def lossless_file(self) -> str:
        return (f'{self.lossless_folder}/'
                f'{self.video.name}_{self.frame.scale}_{self.fps}.mp4')

    @property
    def compressed_folder(self) -> str:
        folder = f'{self.project}/{self._compressed_folder}/{self.basename}'
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def compressed_file(self):
        return f'{self.compressed_folder}/tile{self.tile.id}.mp4'

    @property
    def segment_folder(self):
        folder = f'{self.project}/{self._segment_folder}/{self.basename}'
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def segment_file(self):
        return f'{self.segment_folder}/tile{self.tile.id}_{self.chunk:03}.mp4'

    @property
    def dectime_folder(self):
        folder = f'{self.project}/{self._dectime_folder}/{self.basename}'
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def dectime_log(self):
        return f'{self.dectime_folder}/tile{self.tile.id}_{self.chunk:03}.log'

    @property
    def dectime_raw_json(self):
        filename = f'{self.project}/dectime_raw.json'
        return filename


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
