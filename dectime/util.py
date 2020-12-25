import json
import os
from typing import Union


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


class AutoDict(dict):
    def __init__(self, return_type='dict'):
        super().__init__()
        self.return_type = return_type

    def __missing__(self, key):
        value = self[key] = eval(f'{self.return_type}()')
        return value


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


class Pattern:
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


class Config:
    def __init__(self, config: str):

        with open(f'{config}', 'r') as f:
            self.config_data = json.load(f)

        self.project = self.config_data['project']
        self.factor = self.config_data['factor']
        self.frame = Frame(self.config_data['scale'])
        self.fps = self.config_data['fps']
        self.gop = self.config_data['gop']

        self.quality_list = self.config_data['quality_list']
        self.videos_list = []
        self.pattern_list = []

        self._videos_list()
        self._pattern_list()
        print()

    def _videos_list(self):
        videos_list = self.config_data['videos_list']
        for name in videos_list:
            video_tuple = Video(name, videos_list[name])
            self.videos_list.append(video_tuple)

    def _pattern_list(self):
        pattern_list = self.config_data['tile_list']

        for pattern_str in pattern_list:
            pattern = Pattern(pattern_str, self.frame)
            self.pattern_list.append(pattern)


class Factors:
    _video: Union[Video, None] = None
    _pattern = Pattern('1x1', Frame('1x1'))
    _quality = 28
    _tile = _pattern.tiles_list[0]
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
    def pattern(self) -> Pattern:
        assert self._pattern is not None, "The pattern variable is not defined."
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        self._pattern = value

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
        self.pattern = None
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
                f'{self.pattern.pattern}_'
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
        return f'{self.segment_folder}/tile{self.tile}_{self.chunk:03}.mp4'

    @property
    def dectime_folder(self):
        folder = f'{self.project}/{self._dectime_folder}/{self.basename}'
        os.makedirs(folder, exist_ok=True)
        return folder

    @property
    def dectime_file(self):
        return f'{self.dectime_folder}/tile{self.tile}_{self.chunk:03}.mp4'


class VideoState(Paths):
    def __init__(self, config: str,
                 original_folder='original',
                 lossless_folder='lossless',
                 compressed_folder='compressed',
                 segment_folder='segment',
                 dectime_folder='dectime'):
        """
        Class to creat tile files path to process.
        :param config: Config file.
        :param original_folder: Folder that contain the original files.
        :param lossless_folder: Folder to put intermediate lossless
        full-frame video.
        :param compressed_folder: Folder to put compressed tiles.
        :param segment_folder: Folder to put the segments of tiles.
        :param dectime_folder: Folder to put decode log.
        """
        self.config = Config(config)
        self.project = f'results/{self.config.project}'

        self.frame = self.config.frame
        self.fps = self.config.fps
        self.gop = self.config.gop
        self.factor = self.config.factor

        self.videos_list: list = self.config.videos_list
        self.quality_list: list = self.config.quality_list
        self.pattern_list: list = self.config.pattern_list

        self._original_folder = original_folder
        self._lossless_folder = lossless_folder
        self._compressed_folder = compressed_folder
        self._segment_folder = segment_folder
        self._dectime_folder = dectime_folder
