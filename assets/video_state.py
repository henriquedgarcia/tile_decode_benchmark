import os
from typing import Union

from assets.config import Config
from assets.dectime_types import Frame, Tile, Tiling, Video


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


class VideoState(Paths):
    def __init__(self, config: Config):
        """
        Class to creat tile files path to process.
        :param config: Config object.
        original_folder: Folder that contain the original files.
        lossless_folder: Folder to put intermediate lossless
        full-frame video.
        compressed_folder: Folder to put compressed tiles_list.
        segment_folder: Folder to put the segments of tiles_list.
        dectime_folder: Folder to put decode log.
        """
        self.project = f'results/{config.project}'

        self.frame = config.frame
        self.fps = config.fps
        self.gop = config.gop
        self.factor = config.rate_control

        self.videos_list: list = config.videos_list
        self.quality_list: list = config.quality_list
        self.pattern_list: list = config.tiling_list

        self._original_folder = config.original_folder
        self._lossless_folder = config.lossless_folder

        self._compressed_folder = config.compressed_folder
        self._segment_folder = config.segment_folder
        self._dectime_folder = config.dectime_folder
