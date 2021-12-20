from itertools import product as prod
from pathlib import Path
from typing import Any, Dict, List, Union, NamedTuple, Tuple, Callable, Optional
from lib.util import splitx
from logging import debug
import json


class Position(NamedTuple):
    x: float
    y: float
    z: float = None

    def __str__(self):
        string = f'({self.x}, {self.y}'
        if self.z is not None:
            string = string + f',{self.z}'
        string = string + ')'
        return string


class Resolution:
    w: int
    h: int
    _resolution: str
    shape: Tuple[int]

    def __init__(self, resolution: Union[str, tuple]):
        if isinstance(resolution, str):
            self.resolution = resolution
        elif isinstance(resolution, tuple):
            self.shape = resolution

    @property
    def resolution(self) -> str:
        self._resolution = f'{self.w}x{self.h}'
        return self._resolution

    @resolution.setter
    def resolution(self, res: str):
        self._resolution = res
        self.w, self.h = tuple(map(int, res.split('x')))

    @property
    def shape(self) -> tuple:
        return self.h, self.w

    @shape.setter
    def shape(self, shape: tuple):
        self.h, self.w = shape

    def __iter__(self):
        return iter((self.h, self.w))

    def __str__(self):
        return self._resolution

class Chunk:
    duration: int
    id: int

    def __init__(self, duration, id):
        self.duration = duration
        self.id = id

    def __str__(self):
        return str(self.id)

    def __repr__(self):
        return NamedTuple('Chunk', [('id', int), ('duration', str)]
                          )(self.id, f'{self.duration} s')


class Frame:
    def __init__(self, resolution: Resolution,
                 position: Position = None):
        """
        :param resolution: scale: a string like "1200x600" or a shape numpy-like (y,x)
        """
        self.resolution = resolution
        self.position = position

    def __repr__(self):
        return f'{self.resolution}@{self.position}'

    def __iter__(self):
        iter((self.resolution, self.position))

    def __str__(self):
        return self.resolution

    @property
    def x(self):
        return self.position.x

    @property
    def y(self):
        return self.position.y

    @property
    def h(self):
        return self.resolution.h

    @property
    def w(self):
        return self.resolution.w


class Tile:
    # todo: implementar a seguinte ideia. Um tile é um objeto que possui um frame e está contido em um frame maior.
    # todo: O Frame contem informações espaciais da imagem, como dimensão, informação de cor e opcionalmente da projeção
    # todo: o Tile terá informações temporais e de posição, como idx, posição e Tiling.
    # todo: O Tiling terá informações sobre o frame da projeção, total de tiles, segmentação MxN, e uma referência pra cada tile. (lista de tiles)

    def __init__(self, idx: int, frame: Frame):
        self.frame = frame
        self.idx = idx

    def __str__(self):
        return str(self.idx)


class Tiling:
    _pattern_list: List[str]
    _pattern: str
    tile_frame: Frame
    total_tiles: int
    _tiles_list: List[Tile] = None
    n: int

    def __init__(self, pattern: str, frame: Frame):
        self.pattern = pattern
        self.frame = frame

    @property
    def tiles_list(self):
        if self._tiles_list is not None:
            return self._tiles_list

        resolution = Resolution((self.frame.h // self.n,
                                 self.frame.w // self.m))

        pos_iterator = prod(range(0, self.frame.h, resolution.h),
                            range(0, self.frame.w, resolution.w))
        tiles_list = []
        for idx, (y, x) in enumerate(pos_iterator):
            pos = Position(x, y)
            tile = Tile(idx, Frame(resolution, pos))
            tiles_list.append(tile)
            self._tiles_list = tiles_list
        return tiles_list

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self._pattern = pattern
        self.m, self.n = splitx(pattern)
        self.total_tiles = self.m * self.n

    def __str__(self):
        return self.pattern


class Config:
    config_data: dict = {}

    def __init__(self, config_file: str):
        debug(f'Loading {config_file}.')
        
        self.config_file = Path(config_file)
        content = self.config_file.read_text(encoding='utf-8')
        self.config_data = json.loads(content)

        videos_filename = self.config_data['videos_file']
        videos_file = Path(f'config/{videos_filename}')
        content = videos_file.read_text()
        video_list = json.loads(content)
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
    quality: int = None
    tile: Tile = None
    chunk: int = None

    # Characteristics
    fps: int
    gop: int
    frame: Frame
    original: str
    offset: str
    duration: int
    chunks: range
    group: int

    # Lists
    names_list = []
    tiling_list = []
    tiles_list = {}
    quality_list = []
    chunk_list = {}

    def __str__(self):
        return self.state_str

    def __init__(self, config: Config, deep: int):
        self.config = config
        self.deep = deep

        self.project = Path('results') / config['project']
        self.result_error_metric: str = config['project']
        self.decoding_num: int = config['decoding_num']
        self.codec: str = config['codec']
        self.codec_params: str = config['codec_params']
        self.distributions: list[str] = config['distributions']
        self.rate_control: str = config['rate_control']
        self.original_quality: str = config['original_quality']

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
            return None

        for self.name in self.config.videos_list:
            if deep == 1:
                count += 1
                yield count
                continue
            for pattern in self.config['tiling_list']:
                self.tiling = Tiling(pattern, self.frame)
                if deep == 2:
                    count += 1
                    yield count
                    continue
                for self.quality in self.config['quality_list']:
                    if deep == 3:
                        count += 1
                        yield count
                        continue
                    for self.tile in self.tiling.tiles_list:
                        if deep == 4:
                            count += 1
                            yield count
                            continue
                        for self.chunk in self.chunks:
                            if deep == 5:
                                count += 1
                                yield count
                                continue

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        video_info: dict = self.config.videos_list[name]

        self.original: str = video_info['original']
        self.frame = Frame(Resolution(video_info['scale']), Position(0, 0))
        self.projection: str = video_info['projection']
        self.offset = str(video_info['offset'])
        self.duration = int(video_info['duration'])
        self.group = int(video_info['group'])
        self.fps = int(video_info['fps'])
        self.gop = int(video_info['gop'])
        self.chunks = range(1, (int(self.duration / (self.gop / self.fps)) + 1))

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    @tiling.setter
    def tiling(self, tiling: Union[str, Tiling]):
        if isinstance(tiling, Tiling):
            self._tiling = tiling
        elif isinstance(tiling, str):
            self._tiling = Tiling(tiling, self.frame)

    @property
    def factors_list(self) -> list:
        factors = []
        if self.name is not None:
            factors.append(self.name)
        # if self.projection is not None:
        #     factors.append(self.projection)
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
    def state_str(self) -> str:
        state_str = '_'.join(map(str, self.factors_list))
        return state_str

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
    def basename(self):
        # todo: remover essa gambiarra na próxima rodada
        name = self.name.replace("cmp_", "")
        name = name.replace("erp_", "")
        return Path(f'{name}_'
                    f'{self.frame.resolution}_'
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
        return folder / f'{self.name}_{self.frame}_{self.fps}.mp4'

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
                        f'{self.projection}_'
                        f'{self.frame}_'
                        f'{self.fps}_'
                        f'{self.tiling}_'
                        f'{self.rate_control}{self.original_quality}')

        folder = self.project / self.compressed_folder / basename
        return folder / f'tile{self.tile}.mp4'

    @property
    def quality_csv(self) -> Union[Path, None]:
        folder = self.project / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return (folder
                / f'tile{self.tile}_psnr.csv')

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
