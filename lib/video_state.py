import json
from dataclasses import dataclass
from logging import debug
from pathlib import Path
from typing import (Any, Dict, List, Union, Optional)

import numpy as np

from assets import Resolution, Position
from lib.util import splitx

global config
global state


@dataclass
class Frame:
    image: Optional[np.ndarray] = None
    info: Optional[dict] = None
    resolution: Optional[Resolution] = None


class Chunk:
    def __init__(self, idx, duration: float, n_frames: int,
                 sequence: List[Frame] = None):
        """ um conjunto de frames"""
        self.duration = duration
        self.idx = idx
        self.n_frames = n_frames
        self.sequence = sequence

    def __str__(self):
        return str(self.idx)


class Tile:
    def __init__(self, idx: int,
                 resolution: Resolution,
                 position: Position,
                 chunk_list: List[Chunk] = None):
        """
        Um tile contem uma ID, uma resolução, uma posição e conjunto de chunks.

        :param idx:
        :param resolution:
        :param position:
        :param chunk_list:
        """
        self.idx = idx
        self.resolution = resolution
        self.position = position
        self.chunk_list = chunk_list

    def __str__(self):
        return str(self.idx)


class Tiling:
    _pattern: str
    n_tiles: int
    m: int
    n: int
    _tiles_list: Optional[List[Tile]]

    def __init__(self, pattern: str, proj_res: Resolution):
        """
        A tiling contain a tile pattern, tile resolution and tiling list
        :param pattern:
        :param proj_res:
        """
        self.pattern = pattern
        self.proj_res = proj_res
        self.tile_res = proj_res / self.shape

        self.tiles_list = []
        for n, y in enumerate(range(0, self.proj_res.H, self.tile_res.H)):
            for m, x in enumerate(range(0, self.proj_res.W, self.tile_res.W)):
                N, M = self.shape
                idx = m + n * M
                pos = Position(x, y)
                self.tiles_list.append(Tile(idx, self.tile_res, pos))

    @property
    def tiles_list(self):
        return self._tiles_list

    @tiles_list.setter
    def tiles_list(self, tiles_list: List[Tile]):
        self._tiles_list = tiles_list

    @property
    def pattern(self) -> str:
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self._pattern = pattern
        self.n, self.m = splitx(pattern)
        self.n_tiles = self.m * self.n

    @property
    def shape(self) -> tuple:
        return self.n, self.m

    def __str__(self):
        return self.pattern


class Config:
    config_data: dict = {}

    def __init__(self, config_file: str):
        debug(f'Loading {config_file}.')

        self.config_file = Path(config_file)

        with self.config_file.open('r', encoding='utf-8') as f:
            self.config_data = json.load(f)

        videos_file_json = self.config_data['videos_file']
        videos_file_json = Path(f'config/{videos_file_json}')

        with videos_file_json.open('r', encoding='utf-8') as f:
            video_list = json.load(f)
            video_list = video_list['videos_list']

        for name in video_list:
            fps = self.config_data['fps']
            gop = self.config_data['gop']
            video_list[name].update({"fps": fps, "gop": gop})

        self.videos_list: Dict[str, Any] = video_list

    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value


class VideoContext:
    # Folders
    original_folder = Path('original')
    lossless_folder = Path('lossless')
    compressed_folder = Path('compressed')
    segment_folder = Path('segment')
    dectime_folder = Path('dectime')
    stats_folder = Path('stats')
    graphs_folder = Path("graphs")
    siti_folder = Path("siti")
    _check_folder = Path('check')
    quality_folder = Path('quality')

    # Factors
    _name: str = None
    projection: str = None
    _tiling: Tiling = None
    _quality: int = None
    _tile: Tile = None
    _chunk: int = None

    # Characteristics
    fps: int
    gop: int
    resolution: Resolution
    original: str
    offset: str
    duration: int
    chunks: range
    n_chunks: int
    chunk_dur: float
    group: int

    # Lists
    _names_list = []
    _tiling_list = []
    _tiles_list = []
    _quality_list = []
    _chunks_list = []

    def __init__(self, conf: Config, deep: int):
        global config
        config = conf
        self.deep = deep
        self.project = Path('results') / config['project']
        self.result_error_metric: str = config['project']
        self.decoding_num: int = config['decoding_num']
        self.codec: str = config['codec']
        self.codec_params: str = config['codec_params']
        self.distributions: list[str] = config['distributions']
        self.rate_control: str = config['rate_control']
        self.original_quality: str = config['original_quality']

        self.names_list = list(config['videos_list'].keys())
        self.tiling_list = config['tiling_list']
        self.quality_list = config['quality_list']

    def __str__(self):
        return self.state_str

    def __len__(self):
        i = 0
        for i in self: ...
        return i

    def __iter__(self):
        count = 0
        deep = self.deep
        if deep == 0:
            count += 1
            yield count
            return
        for self.name in self.names_list:
            if deep == 1:
                count += 1
                yield count
                continue
            for self.tiling in self.tiling_list:
                if deep == 2:
                    count += 1
                    yield count
                    continue
                for self.quality in self.quality_list:
                    if deep == 3:
                        count += 1
                        yield count
                        continue
                    for self.tile in self.tiles_list:
                        if deep == 4:
                            count += 1
                            yield count
                            continue
                        for self.chunk in self.chunks_list:
                            if deep == 5:
                                count += 1
                                yield count
                                continue

    def make_name(self, base_name: Union[str, None] = None,
                  ext: str = None, other: Any = None,
                  separator='_') -> str:
        name = self.state_str.replace('_', separator)
        if base_name:
            name = f'{base_name}{separator}{name}'
        if other:
            name = f'{name}{separator}{other}'
        if ext:
            name = f'{name}.{ext}'
        return name

    @property
    def names_list(self) -> list:
        return self._names_list

    @names_list.setter
    def names_list(self, videos_list: list[str]):
        self._videos_list = videos_list

    @property
    def tiling_list(self) -> List[str, ...]:
        return self._tiling_list

    @tiling_list.setter
    def tiling_list(self, tiling_list: List[str]):
        self._tiling_list = tiling_list

    @property
    def quality_list(self) -> List[int]:
        return self._quality_list

    @quality_list.setter
    def quality_list(self, quality_list: List[int]):
        self._quality_list = quality_list

    @property
    def tiles_list(self) -> List[Tile]:
        return self._tiles_list

    @tiles_list.setter
    def tiles_list(self, tiles_list: List[Tile]):
        self._tiles_list = tiles_list

    @property
    def chunks_list(self) -> List[Chunk]:
        return self._chunks_list

    @chunks_list.setter
    def chunks_list(self, chunks_list: (str, Tiling)):
        self._chunks_list = chunks_list

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        video_info: dict = config.videos_list[name]

        self.original: str = video_info['original']
        self.resolution = Resolution(video_info['scale'])
        self.projection: str = video_info['projection']
        self.offset = str(video_info['offset'])
        self.duration = int(video_info['duration'])
        self.group = int(video_info['group'])
        self.fps = int(video_info['fps'])
        self.gop = int(video_info['gop'])
        self.chunk_dur = (self.gop / self.fps)
        self.n_chunks = int(self.duration / self.chunk_dur)
        self.chunks_list = [Chunk(idx, self.chunk_dur, n_frames=self.gop) for idx in
                            range(1, self.n_chunks + 1)]
        global state
        state = self.state

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    @tiling.setter
    def tiling(self, tiling: (str, Tiling)):
        if isinstance(tiling, Tiling):
            self._tiling = tiling
        elif isinstance(tiling, str):
            self._tiling = Tiling(tiling, self.resolution)
        self.tiles_list = self._tiling.tiles_list
        global state
        state = self.state

    @property
    def quality(self) -> int:
        return self._quality

    @quality.setter
    def quality(self, quality: int):
        self._quality = quality
        global state
        state = self.state

    @property
    def tile(self) -> Tile:
        return self._tile

    @tile.setter
    def tile(self, tile: Tile):
        self._tile = tile
        global state
        state = self.state

    @property
    def chunk(self) -> int:
        return self._chunk

    @chunk.setter
    def chunk(self, chunk: int):
        self._chunk = chunk
        global state
        state = self.state

    @property
    def factors_list(self) -> list:
        factors = []
        if self.name is not None:
            factors.append(self.name)
        # if self.projection is not None:
        #    factors.append(self.projection)
        if self.tiling is not None:
            factors.append(str(self.tiling))
        if self.quality is not None:
            factors.append(str(self.quality))
        if self.tile is not None:
            factors.append(str(self.tile))
        if self.chunk is not None:
            factors.append(str(self.chunk))
        return factors

    @property
    def state(self):
        return self.factors_list

    @property
    def state_str(self) -> str:
        factors = []
        if self.name is not None:
            factors.append(self.name)
        # if self.projection is not None:
        #     factors.append(self.projection)
        if self.tiling is not None:
            factors.append(str(self.tiling))
        if self.quality is not None:
            factors.append(f'{self.rate_control}{self.quality}')
        if self.tile is not None:
            factors.append(f'tile{self.tile}')
        if self.chunk is not None:
            factors.append(f'chunk{self.chunk}')
        state_str = '_'.join(factors)
        return state_str

    @property
    def basename(self):
        # todo: remover essa gambiarra na próxima rodada
        name = self.name.replace("cmp_", "")
        name = name.replace("erp_", "")
        return Path(f'{name}_'
                    f'{self.resolution}_'
                    f'{self.fps}_'
                    f'{self.tiling}_'
                    f'{self.rate_control}{self.quality}')

    @property
    def original_file(self) -> Path:
        return self.original_folder / self.original

    @property
    def lossless_file(self) -> Path:
        folder = self.project / self.lossless_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.name}_{self.resolution}_{self.fps}.mp4'

    @property
    def compressed_file(self) -> Path:
        folder = self.project / self.compressed_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def segment_file(self) -> Path:
        folder = self.project / self.segment_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}_{self.chunk:03}.mp4'

    @property
    def dectime_log(self) -> Path:
        folder = self.project / self.dectime_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}_{self.chunk:03}.log'

    @property
    def dectime_json_file(self) -> Path:
        folder = self.project / self.dectime_folder
        return folder / 'dectime.json'

    @property
    def siti_results(self) -> Path:
        folder = self.project / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.name}_siti_results.csv'

    @property
    def siti_movie(self) -> Path:
        folder = self.project / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.name}_siti_movie.mp4'

    @property
    def siti_stats(self) -> Path:
        folder = self.project / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.name}_siti_stats.json'

    @property
    def reference_file(self) -> Path:
        basename = Path(f'{self.name}_'
                        # f'{self.projection}_'
                        f'{self.resolution}_'
                        f'{self.fps}_'
                        f'{self.tiling}_'
                        f'{self.rate_control}{self.original_quality}')

        folder = self.project / self.compressed_folder / basename
        return folder / f'tile{self.tile}.mp4'

    @property
    def quality_pickle(self) -> Union[Path, None]:
        folder = self.project / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return (folder
                / f'Quality_metrics.pickle')

    @property
    def quality_csv(self) -> Union[Path, None]:
        folder = self.project / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return (folder
                / f'tile{self.tile}.csv')

    @property
    def quality_result_json(self) -> Union[Path, None]:
        return (self.project
                / self.quality_folder
                / 'compressed_quality_result.json')

    @property
    def check_folder(self) -> Path:
        folder = self.project / self._check_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

# class VideoBase:
#     frame: Frame
#     duration: int
#     fps: int
#     gop: int
#     chunks_list: List[Chunk_]


# class Tiling_:
#     tiling: str
#     M: int
#     N: int
#     resolution: Resolution
#     tiles: List[Tile_]
#     total_tiles: int
#     tile_res: Resolution
#
#     def __init__(self, tiling: str, frame: Frame):
#         self.tiling = tiling
#         self.frame = frame
#
#     @property
#     def tiling(self):
#         return self._tiling
#
#     @tiling.setter
#     def tiling(self, tiling: str):
#         self._tiling = tiling
#         self.m, self.n = splitx(tiling)
#         self.total_tiles = self.m * self.n
#
#     def __str__(self):
#         return self.tiling
#
#     @property
#     def tiles(self):
#         h, w = self.frame.size.shape
#         self.tile_scale = Frame(scale=(w // self.m,
#                                        h // self.n))
#         pos_iterator = prod(range(0, h, self.tile_scale.h),
#                             range(0, w, self.tile_scale.w))
#         tiles_list = [Tile(idx, self.tile_scale, (x, y))
#                       for idx, (y, x) in enumerate(pos_iterator)]
#         return tiles_list


# class Video_:
#     file_path: Path
#     original: Path
#     codec: str
#     group: int
#     _name: str
#
#     # spatial
#     projection: str
#     resolution: Resolution
#     quality_factor: str
#     quality: int
#     tiling: Tiling_
#
#     # temporal
#     fps: int
#     gop: int
#     duration: int
#     offset: int
#     chunk_duration: int
#
#     def __init__(self, config: Config):
#
#     @property
#     def name(self):
#         return self._name
#
#     @name.setter
#     def name(self, name: str):
#         self._name = name

# class Factors:
#     video: Video = None
#     projection: str = None
#     tiling: Tiling = None
#     quality: int = None
#     tile: Tile = None
#     chunk: int = None
#     rate_control: str = None
#
#     @property
#     def name(self) -> Union[str, None]:
#         value = self.video.name if self.video else None
#         return value
#
#     @name.setter
#     def name(self, name: str):
#         self.video.name = name
#
#     @property
#     def projection(self) -> Union[str, None]:
#         value = self.video.projection if self.video else None
#         return value
#
#     @projection.setter
#     def projection(self, projection: str):
#         self.video.projection = projection
#
#     @property
#     def scale(self) -> Union[str, None]:
#         value = self.video.scale if self.video else None
#         return value
#
#     @scale.setter
#     def scale(self, scale: str):
#         self.video.scale = scale
#
#     @property
#     def pattern(self) -> Union[str, None]:
#         value = self.tiling.pattern if self.tiling else None
#         return value
#
#     @pattern.setter
#     def pattern(self, pattern: str):
#         self.tiling.pattern = pattern
#
#     @property
#     def tile_id(self) -> Union[int, None]:
#         value = self.tile.idx if self.tile else None
#         return value
#
#     @property
#     def state(self) -> str:
#         state = None
#         if self.name is not None:
#             state = (f'{state}_{self.name}'
#                      if state else f'{self.name}')
#         if self.projection is not None:
#             state = (f'{state}_{self.projection}'
#                      if state else f'{self.projection}')
#         if self.pattern is not None:
#             state = (f'{state}_{self.pattern}'
#                      if state else f'{self.pattern}')
#         if self.quality is not None:
#             state = (f'{state}_{self.rate_control}{self.quality}'
#                      if state else f'{self.rate_control}{self.quality}')
#         if self.tile is not None:
#             state = (f'{state}_tile{self.tile_id}'
#                      if state else f'tile{self.tile_id}')
#         if self.chunk is not None:
#             state = (f'{state}_chunk{self.chunk}'
#                      if state else f'chunk{self.chunk}')
#         return state
#
#     def get_factors(self):
#         factors = []
#         if self.name is not None:
#             factors.append(self.name)
#         if self.projection is not None:
#             factors.append(self.projection)
#         if self.scale is not None:
#             factors.append(self.scale)
#         if self.pattern is not None:
#             factors.append(self.pattern)
#         if self.quality is not None:
#             factors.append(self.quality)
#         if self.tile_id is not None:
#             factors.append(self.tile_id)
#         if self.chunk is not None:
#             factors.append(self.chunk)
#         return factors
#
#     def get_name(self, base_name: Union[str, None] = None,
#                  ext: Union[str, None] = None,
#                  other: Any = None,
#                  separator='_'):
#         name = self.state.replace('_', separator)
#         if base_name:
#             name = f'{base_name}{separator}{name}'
#         if other:
#             name = f'{name}{separator}{other}'
#         if ext:
#             name = f'{name}.{ext}'
#         return name

# class Params:
#     """
#     Interface
#     ---------
#     This interface represent the constants values for all simulation.
#     """
#     project: Path
#
#     rate_control: str
#     # videos_dict: dict

#
# class Paths(Params, Factors):
#     """
#     original_folder: Folder that contain the original files.
#     lossless_folder: Folder to put intermediate lossless full-frame video.
#     compressed_folder: Folder to put compressed tiles_list.
#     segment_folder: Folder to put the segments of tiles_list.
#     dectime_folder: Folder to put decode log.
#     """
#     _original_folder: Path
#     _lossless_folder: Path
#     _compressed_folder: Path
#     _segment_folder: Path
#     _dectime_folder: Path
#     _siti_folder: Path
#     _check_folder: Path
#
#     @property
#     def basename(self):
#         return Path(f'{self.video.name}_'
#                     f'{self.video.projection}_'
#                     f'{self.video.scale}_'
#                     f'{self.fps}_'
#                     f'{self.pattern}_'
#                     f'{self.rate_control}{self.quality}')
#
#     @property
#     def original_folder(self) -> Path:
#         return self._original_folder
#
#     @property
#     def check_folder(self) -> Path:
#         folder = self.project / self._check_folder
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def lossless_folder(self) -> Path:
#         folder = self.project / self._lossless_folder
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def segment_folder(self) -> Path:
#         folder = self.project / self._segment_folder / self.basename
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def dectime_folder(self) -> Path:
#         folder = self.project / self._dectime_folder / self.basename
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def siti_folder(self) -> Path:
#         folder = self.project / self._siti_folder
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def original_file(self) -> Path:
#         return self.original_folder / self.video.original
#
#     @property
#     def lossless_file(self) -> Path:
#         return self.lossless_folder / f'{self.name}_{self.scale}_{self.fps}.mp4'
#
#     @property
#     def lossless_log(self) -> Path:
#         return self.lossless_file.with_suffix('.log')
#
#     @property
#     def compressed_folder(self) -> Path:
#         folder = self.project / self._compressed_folder / self.basename
#         folder.mkdir(parents=True, exist_ok=True)
#         return folder
#
#     @property
#     def compressed_file(self) -> Path:
#         return self.compressed_folder / f'tile{self.tile_id}.mp4'
#
#     @property
#     def compressed_log(self) -> Path:
#         return self.compressed_file.with_suffix('.log')
#
#     @property
#     def compressed_reference(self) -> Path:
#         basename = Path(f'{self.video.name}_'
#                         f'{self.scale}_'
#                         f'{self.fps}_'
#                         f'{self.pattern}_'
#                         f'{self.rate_control}{self.original_quality}')
#
#         folder = self.project / self._compressed_folder / basename
#         path = folder / f'tile{self.tile_id}.mp4'
#         return path
#
#     @property
#     def compressed_quality_csv(self) -> Union[Path, None]:
#         return self.compressed_folder / f'tile{self.tile_id}.csv'
#
#     @property
#     def compressed_quality_result_json(self) -> Union[Path, None]:
#         return self.project / 'compressed_quality_result.json'
#
#     @property
#     def segment_file(self) -> Path:
#         return self.segment_folder / f'tile{self.tile_id}_{self.chunk:03}.mp4'
#
#     @property
#     def segment_log(self) -> Path:
#         return self.segment_folder / f'tile{self.tile_id}.log'
#
#     @property
#     def dectime_log(self) -> Path:
#         return self.dectime_folder / f'tile{self.tile_id}_{self.chunk:03}.log'
#
#     @property
#     def dectime_json_file(self) -> Path:
#         return self.project / 'dectime.json'
#
#     @property
#     def siti_results(self) -> Path:
#         return self.siti_folder / f'{self.name}_siti_results.csv'
#
#     @property
#     def siti_movie(self) -> Path:
#         return self.siti_folder / f'{self.name}_siti_movie.mp4'
#
#     @property
#     def siti_stats(self) -> Path:
#         return self.siti_folder / f'{self.name}_siti_stats.json'


# class DectimeLists(Params, Factors):
#     _videos_list: List[Video]
#     _names_list: List[str]
#     _pattern_list: List[str]
#     _tiling_list: List[Tiling]
#     _quality_list: List[int]
#     _tile_list: List[int]
#     _chunk_list: List[int]
#     videos_dict: dict
#
#     @property
#     def videos_list(self):
#         return self._videos_list
#
#     @videos_list.setter
#     def videos_list(self, videos_dict: Dict[str, dict]):
#         self._names_list = [name for name in videos_dict]
#         self._videos_list = [Video(name, videos_dict[name])
#                              for name in videos_dict]
#
#     @property
#     def names_list(self):
#         return self._names_list
#
#     @names_list.setter
#     def names_list(self, videos_dict: Dict[str, dict]):
#         self.videos_list = videos_dict
#
#     @property
#     def tiling_list(self):
#         return self._tiling_list
#
#     @tiling_list.setter
#     def tiling_list(self, pattern_list: List[str]):
#         self._pattern_list = pattern_list
#         frame = Frame(self.scale)
#         self._tiling_list = [Tiling(pattern, frame)
#                              for pattern in pattern_list]
#
#     @property
#     def pattern_list(self):
#         return self._pattern_list
#
#     @pattern_list.setter
#     def pattern_list(self, pattern_list: List[str]):
#         self.tiling_list = pattern_list
#
#     @property
#     def quality_list(self):
#         return self._quality_list
#
#     @quality_list.setter
#     def quality_list(self, quality_list: List[int]):
#         self._quality_list = quality_list
#
#     @property
#     def tiles_list(self):
#         return self.tiling.tiles_list
#
#     @property
#     def chunk_list(self):
#         return self.video.chunks


# class AbstractVideoState(ABC, Paths, DectimeLists):
#     deep: int
#
#     @abstractmethod
#     def __init__(self, deep): ...
#
#     def __iter__(self): ...
#
#     def __len__(self):
#         i = 0
#         for i in self: ...
#         return i
#
#     def __str__(self):
#         state = None
#         if self.name is not None:
#             state = (f'{state}_{self.name}'
#                      if state else f'{self.name}')
#         if self.pattern is not None:
#             state = (f'{state}_{self.pattern}'
#                      if state else f'{self.pattern}')
#         if self.quality is not None:
#             state = (f'{state}_{self.rate_control}{self.quality}'
#                      if state else f'{self.rate_control}{self.quality}')
#         if self.tile is not None:
#             state = (f'{state}_tile{self.tile_id}'
#                      if state else f'tile{self.tile_id}')
#         if self.chunk is not None:
#             state = (f'{state}_chunk{self.chunk}'
#                      if state else f'chunk{self.chunk}')
#         return state if state else ''
