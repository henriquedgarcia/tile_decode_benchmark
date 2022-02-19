from dataclasses import dataclass
from itertools import product
from logging import debug
from pathlib import Path
from typing import (Any, Dict, List, Union, Optional)

import matplotlib.pyplot as plt
import numpy as np

from lib.assets import Resolution, Position
from .util import load_json, AutoDict
from .viewport2 import Viewport


@dataclass
class Video:
    name: str
    original: Path
    resolution: Resolution
    projection: str
    offset: str
    duration: int
    group: int
    fps: int
    gop: int
    chunk_dur: float
    n_chunks: int

    def __str__(self):
        return self.name


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
    def __init__(self,
                 resolution: Resolution,
                 position: Position,
                 chunk_list: List[Chunk] = None,
                 idx: int = 0):
        """
        Um tile contem uma ID, uma resolução, uma posição e conjunto de chunks.

        :param idx:
        :param resolution:
        :param position: position on the image
        :param chunk_list:
        """
        self.idx = idx
        self.resolution = resolution
        self.H = resolution.H
        self.W = resolution.W
        self.position = position
        self.x = position.x
        self.y = position.y
        self.chunk_list = chunk_list

    def __str__(self):
        return str(self.idx)

    def __repr__(self):
        return str(f'[{self.idx}]{self.resolution}@{self.position}')
    
    def get_border(self, proj_res: Resolution) -> tuple[tuple, tuple]:
        """

        :param proj_res:
        :return:
        """
        position = self.position
        resolution = self.resolution
        vp = Viewport('1x1')
        vp.resolution = proj_res
        convert = vp.pix2cart

        x_i = position.x  # first row
        x_f = position.x + resolution.W  # last row
        y_i = position.y  # first line
        y_f = position.y + resolution.H  # last line

        xi_xf = range(int(x_i), int(x_f))
        yi_yf = range(int(y_i), int(y_f))

        for x in xi_xf:
            yield (x, y_i), convert(x, y_i)
        for x in xi_xf:
            yield (x, y_f - 1), convert(x, y_f - 1)
        for y in yi_yf:
            yield (x_i, y), convert(x_i, y)
        for y in yi_yf:
            yield (x_f - 1, y), convert(x_f - 1, y)


class Tiling:
    _pattern: Resolution
    _proj_res: Resolution
    _tile_res: Resolution
    n_tiles: int
    _tiles_list: Optional[List[Tile]] = []
    _fov: Resolution

    def __init__(self,
                 pattern: Union[str, Resolution],
                 proj_res: Union[str, Resolution], 
                 fov: str = '1x1'):

        """
        A tiling contain a tile pattern, tile resolution and tiling list
        :param pattern:
        :param proj_res:
        """
        self.pattern = pattern
        self.proj_res = proj_res
        self.tile_res = proj_res / self.pattern.shape
        self.fov = fov
        self.fov_y, self.fov_x = self.fov
        self.viewport = Viewport(f'{fov}')

    def __str__(self):
        return str(self.pattern)

    @property
    def fov(self) -> Resolution:
        return self._fov

    @fov.setter
    def fov(self, value: Union[str, Resolution]):
        if isinstance(value, Resolution):
            self._fov = value
        elif isinstance(value, str):
            self._fov = Resolution(value)

    @property
    def tiles_list(self) -> List[Tile]:
        self._tiles_list = []
        positions = product(
            range(0, self.proj_res.H, self.tile_res.H),
            range(0, self.proj_res.W, self.tile_res.W))
        for idx, (y, x) in enumerate(positions):
            tile = Tile(self.tile_res, Position(x, y), idx=idx)
            self._tiles_list.append(tile)
        return self._tiles_list

    @tiles_list.setter
    def tiles_list(self, tiles_list: List[Tile]):
        self._tiles_list = tiles_list

    @property
    def pattern(self) -> Resolution:
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: Union[str, Resolution]):
        if isinstance(pattern, Resolution):
            self._pattern = pattern
        elif isinstance(pattern, str):
            self._pattern = Resolution(pattern)

        self.n_tiles = self._pattern.W * self._pattern.H

    def get_vptiles(self, position: tuple):
        viewport = self.viewport
        viewport.set_rotation(position)

        tiles = []

        for tile in self.tiles_list:
            for (pixel, point_3d) in tile.get_border(self.proj_res):
                if viewport.is_viewport(point_3d):
                    tiles.append(tile.idx)
                    break

        # self.viewport.project(self.proj_res)
        # for tile in self.tiles_list:
        #     for (pixel, point_3d) in tile.get_border(self.proj_res):
        #         self.viewport.projection[pixel.y, pixel.x] = 100
        # self.viewport.show()
        return tiles

    def draw_borders(self):
        img = np.zeros(self.proj_res.shape)
        for tile in self.tiles_list:
            for pixel, _ in tile.get_border(Resolution('1x1')):
                img[pixel[1], pixel[0]] = 255
        img = img.astype(np.uint8)
        plt.imshow(img)
        plt.show()

    @property
    def proj_res(self) -> Resolution:
        return self._proj_res

    @proj_res.setter
    def proj_res(self, resolution: Union[str, Resolution]):
        if isinstance(resolution, Resolution):
            self._proj_res = resolution
        elif isinstance(resolution, str):
            self._proj_res = Resolution(resolution)


    @property
    def tile_res(self) -> Resolution:
        return self._tile_res

    @tile_res.setter
    def tile_res(self, resolution: Union[str, Resolution]):
        if isinstance(resolution, Resolution):
            self._tile_res = resolution
        elif isinstance(resolution, str):
            self._tile_res = Resolution(resolution)

    @property
    def shape(self) -> tuple:
        return self.pattern.shape

class Config:
    config_data: dict = {}

    def __init__(self, config_file: str):
        debug(f'Loading {config_file}.')

        self.config_file = Path(config_file)

        self.config_data = load_json(self.config_file)

        self.videos_list = load_json(f'config/{self.config_data["videos_file"]}')['videos_list']

        for name in self.videos_list:
            fps = self.config_data['fps']
            gop = self.config_data['gop']
            self.videos_list[name].update({"fps": fps, "gop": gop})

        self.config_data['videos_list'] = self.videos_list

        plot_config = f'config/{self.config_data["plot_config"]}'
        self.plot_config: AutoDict = load_json(plot_config)

    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value


class Factors:
    config: Config
    rate_control: str
    dataset_name: str
    # Factors
    _video: Video = None
    _tiling: Tiling = None
    _quality: int = None
    _tile: Tile = None
    _chunk: Chunk = None

    # Lists
    _videos_list = []
    _tiling_list = []
    _tiles_list = []
    _quality_list = []
    _chunks_list = []

    @property
    def state(self) -> list:
        state = []
        if self.video is not None:
            state.append(str(self.video))
        if self.tiling is not None:
            state.append(str(self.tiling))
        if self.quality is not None:
            state.append(str(self.quality))
        if self.tile is not None:
            state.append(str(self.tile))
        if self.chunk is not None:
            state.append(str(self.chunk))
        return state

    #########
    @property
    def videos_list(self) -> List[Video]:
        self._videos_list = []
        for name in self.config.videos_list:
            video_info: dict = self.config.videos_list[name]
            fps = int(video_info['fps'])
            gop = int(video_info['gop'])
            duration = int(video_info['duration'])
            chunk_dur = (gop / fps)
            video = Video(name=name,
                          original=video_info['original'],
                          resolution=Resolution(video_info['scale']),
                          projection=video_info['projection'],
                          offset=str(video_info['offset']),
                          duration=duration,
                          group=int(video_info['group']),
                          fps=fps,
                          gop=gop,
                          chunk_dur=(gop / fps),
                          n_chunks=int(duration / chunk_dur)
                          )

            self._videos_list.append(video)
        return self._videos_list

    @videos_list.setter
    def videos_list(self, videos_list: List[Video]):
        self._videos_list = videos_list

    @property
    def video(self) -> Video:
        return self._video

    @video.setter
    def video(self, video: Video):
        self._video = video

    #########
    @property
    def tiling_list(self) -> List[Tiling]:
        self._tiling_list = []
        for pattern in self.config['tiling_list']:
            proj_res = self.video.resolution if self.video else pattern
            tiling = Tiling(pattern=pattern,
                            proj_res=proj_res,
                            fov=self.config['fov'])
            self._tiling_list.append(tiling)
        return self._tiling_list

    @tiling_list.setter
    def tiling_list(self, tiling_list: List[Tiling]):
        self._tiling_list = tiling_list

    @property
    def tiling(self) -> Tiling:
        return self._tiling

    @tiling.setter
    def tiling(self, tiling: Tiling):
        self._tiling = tiling

    #########
    @property
    def quality_list(self) -> List[int]:
        self._quality_list = self.config['quality_list']
        return self._quality_list

    @quality_list.setter
    def quality_list(self, quality_list: List[int]):
        self._quality_list = quality_list

    @property
    def quality(self) -> int:
        return self._quality

    @quality.setter
    def quality(self, quality: int):
        self._quality = quality

    #########
    @property
    def tiles_list(self) -> List[Tile]:
        self._tiles_list = self.tiling.tiles_list
        return self._tiles_list

    @tiles_list.setter
    def tiles_list(self, tiles_list: List[Tile]):
        self._tiles_list = tiles_list

    @property
    def tile(self) -> Tile:
        return self._tile

    @tile.setter
    def tile(self, tile: Tile):
        self._tile = tile

    #########
    @property
    def chunks_list(self) -> List[Chunk]:
        self._chunks_list = []
        for idx in range(1, self.video.n_chunks + 1):
            chunk = Chunk(idx, self.video.chunk_dur,
                          n_frames=self.video.gop)
            self._chunks_list.append(chunk)
        return self._chunks_list

    @chunks_list.setter
    def chunks_list(self, chunks_list: List[Chunk]):
        self._chunks_list = chunks_list

    @property
    def chunk(self) -> Chunk:
        return self._chunk

    @chunk.setter
    def chunk(self, chunk: Chunk):
        self._chunk = chunk


class ProjectPaths(Factors):
    # Folders
    original_folder = Path('original')
    lossless_folder = Path('lossless')
    compressed_folder = Path('compressed')
    segment_folder = Path('segment')
    dectime_folder = Path('dectime')
    stats_folder = Path('stats')
    _graphs_folder = Path('graphs')
    siti_folder = Path('siti')
    _check_folder = Path('check')
    quality_folder = Path('quality')
    get_tiles_folder = Path('get_tiles')

    @property
    def basename(self):
        return Path(f'{self.video.name}_'
                    f'{self.video.resolution}_'
                    f'{self.video.fps}_'
                    f'{self.tiling}_'
                    f'{self.rate_control}{self.quality}')

    @property
    def project_path(self) -> Path:
        return Path('results') / self.config['project']

    @property
    def original_file(self) -> Path:
        return self.original_folder / self.video.original

    @property
    def lossless_file(self) -> Path:
        folder = self.project_path / self.lossless_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video.name}_{self.video.resolution}_{self.video.fps}.mp4'

    @property
    def compressed_file(self) -> Path:
        folder = self.project_path / self.compressed_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def segment_file(self) -> Path:
        folder = self.project_path / self.segment_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.mp4'

    @property
    def dectime_log(self) -> Path:
        folder = self.project_path / self.dectime_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.log'

    @property
    def dectime_json_file(self) -> Path:
        folder = self.project_path / self.dectime_folder
        return folder / 'dectime.json'

    @property
    def siti_results(self) -> Path:
        folder = self.project_path / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video.name}_siti_results.csv'

    @property
    def siti_movie(self) -> Path:
        folder = self.project_path / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video.name}_siti_movie.mp4'

    @property
    def siti_stats(self) -> Path:
        folder = self.project_path / self.siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video.name}_siti_stats.json'

    @property
    def reference_file(self) -> Path:
        basename = Path(f'{self.video.name}_'
                        f'{self.video.resolution}_'
                        f'{self.video.fps}_'
                        f'{self.tiling}_'
                        f'{self.config["rate_control"]}{self.config["original_quality"]}')

        folder = self.project_path / self.compressed_folder / basename
        return folder / f'tile{self.tile}.mp4'

    @property
    def quality_video_csv(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.csv'

    @property
    def quality_result_pickle(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'Quality_metrics.pickle'

    @property
    def quality_result_json(self) -> Union[Path, None]:
        return (self.project_path
                / self.quality_folder
                / 'compressed_quality_result.json')

    @property
    def check_folder(self) -> Path:
        folder = self.project_path / self._check_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def graphs_folder(self) -> Path:
        folder = self.project_path / self._graphs_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def get_tiles_pickle(self) -> Path:
        folder = self.project_path / self.get_tiles_folder
        folder.mkdir(parents=True, exist_ok=True)
        name = str(self.video).replace('_cmp', '')
        name = str(name).replace('_erp', '')
        filename = f'get_tiles_{self.dataset_name}_{name}_{self.video.projection}_{self.tiling}.pickle'
        return folder / filename

    @property
    def dataset_pickle(self) -> Path:
        self.database_folder = Path('datasets')
        self.database_path = self.database_folder / self.dataset_name
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.database_pickle = self.database_path / f'{self.dataset_name}.pickle'
        return self.database_pickle


class VideoContext(ProjectPaths):
    def __init__(self, conf: Config, deep: int):
        self.config = conf
        
        self.deep = deep
        self.error_metric: str = self.config['error_metric']
        self.decoding_num: int = self.config['decoding_num']
        self.codec: str = self.config['codec']
        self.codec_params: str = self.config['codec_params']
        self.distributions: List[str] = self.config['distributions']
        self.rate_control: str = self.config['rate_control']
        self.original_quality: str = self.config['original_quality']

    def __str__(self):
        factors = []
        if self.video is not None:
            factors.append(f'{self.video}')
        if self.tiling is not None:
            factors.append(f'{self.tiling}')
        if self.quality is not None:
            factors.append(f'{self.rate_control}{self.quality}')
        if self.tile is not None:
            factors.append(f'tile{self.tile}')
        if self.chunk is not None:
            factors.append(f'chunk{self.chunk}')
        ctx_str = '_'.join(factors)
        return ctx_str

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

        for self.video in self.videos_list:
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
        name = str(self).replace('_', separator)
        if base_name:
            name = f'{base_name}{separator}{name}'
        if other:
            name = f'{name}{separator}{other}'
        if ext:
            name = f'{name}.{ext}'
        return name
