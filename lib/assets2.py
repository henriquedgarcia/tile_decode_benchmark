from abc import ABC
import numpy as np
from collections  import defaultdict
import json
from pathlib import Path
from enum import Enum
import pandas as pd
from typing import Union, Callable
from .util2 import splitx
import matplotlib.pyplot as plt


class Config:
    config_data: dict

    def __init__(self, config_file: Path):
        print(f'Loading {config_file}.')

        self.config_file = config_file
        self.config_data = json.loads(config_file.read_text())

        videos_file = Path('config/' + self.config_data["videos_file"])
        videos_list = json.loads(videos_file.read_text())
        self.videos_list = videos_list['videos_list']

        for name in self.videos_list:
            self.videos_list[name].update({"fps": self.config_data['fps'],
                                           "gop": self.config_data['gop']})

        self.config_data['videos_list'] = self.videos_list

    def __getitem__(self, key):
        return self.config_data[key]

    def __setitem__(self, key, value):
        self.config_data[key] = value


class Factors:
    bins: Union[int, str] = None
    video: str = None
    quality_ref: str = '0'
    quality: str = None
    tiling: str = None
    metric: str = None
    tile: str = None
    chunk: str = None
    proj: str = None
    _name_list: list[str]
    config: Config
    user: int

    # <editor-fold desc="Main lists">
    @property
    def videos_list(self) -> dict[str, dict[str, Union[int, float, str]]]:
        return self.config.videos_list

    @property
    def name_list(self) -> list[str]:
        return list(set([video.replace('_cmp', '').replace('_erp', '') for video in self.videos_list]))

    @property
    def proj_list(self) -> list[str]:
        projs = set([self.videos_list[video]['projection'] for video in self.videos_list])
        return list(projs)

    @property
    def tiling_list(self) -> list[str]:
        return self.config['tiling_list']

    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        return quality_list

    @property
    def tile_list(self) -> list[str]:
        tile_m, tile_n = map(int, self.tiling.split('x'))
        return list(map(str, range(tile_n * tile_m)))

    @property
    def chunk_list(self) -> list[str]:
        return list(map(str, range(1, int(self.duration) + 1)))

    @property
    def frame_list(self) -> list[str]:
        return list(range(int(self.duration*int(self.fps))))


    # </editor-fold>

    # <editor-fold desc="Video Property">
    @property
    def name(self) -> str:
        name = self.video.replace('_cmp', '').replace('_erp', '')
        return name

    @property
    def vid_proj(self) -> str:
        return self.videos_list[self.video]['projection']

    @property
    def resolution(self) -> str:
        return self.videos_list[self.video]['scale']

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.videos_list[self.video]['scale'])
        return h, w, 3

    @property
    def fps(self) -> str:
        return self.config['fps']

    @property
    def gop(self) -> str:
        return self.config['gop']

    @property
    def duration(self) -> str:
        return self.videos_list[self.video]['duration']

    @property
    def offset(self) -> int:
        return int(self.videos_list[self.video]['offset'])

    @property
    def chunk_dur(self) -> int:
        return int(self.gop) // int(self.fps)

    @property
    def original(self) -> str:
        return self.videos_list[self.video]['original']
    # </editor-fold>

    # Tile Decoding Benchmark
    @property
    def decoding_num(self) -> int:
        return int(self.config['decoding_num'])

    # Metrics
    @property
    def metric_list(self) -> list[str]:
        return ['time', 'rate', 'time_std', 'PSNR', 'WS-PSNR', 'S-PSNR']

    # GetTiles
    @property
    def fov(self) -> str:
        return self.config['fov']


class GlobalPaths(Factors, ABC):
    dataset_name = 'nasrabadi_28videos'
    worker_name:str = None
    overwrite = False
    dectime_folder = Path('dectime')
    graphs_folder = Path('graphs')
    get_tiles_folder = Path('get_tiles')
    _workfolder: Path = None

    @property
    def workfolder(self) -> Path:
        """
        Need None
        set _workfolder=None if locked
        :return:
        """
        if self._workfolder is None:
            folder = self.project_path / self.worker_name / f'{self.__class__.__name__}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        else:
            return self._workfolder

    @workfolder.setter
    def workfolder(self, value):
        self._workfolder = value

    @property
    def project_path(self) -> Path:
        return Path('results') / self.config['project']

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.__class__.__name__}\n'
              f'  project: {self.project_path}\n'
              f'  codec: {self.config["codec"]}\n'
              f'  fps: {self.config["fps"]}\n'
              f'  gop: {self.config["gop"]}\n'
              f'  qualities: {self.config["quality_list"]}\n'
              f'  patterns: {self.config["tiling_list"]}'
              )
        print('=' * 70)

    @staticmethod
    def count_decoding(dectime_log: Path) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :return:
        """
        try:
            content = dectime_log.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            print('ERROR: UnicodeDecodeError. Cleaning.')
            dectime_log.unlink()
            return 0
        except FileNotFoundError:
            print('ERROR: FileNotFoundError. Return 0.')
            return 0

        return len(['' for line in content if 'utime' in line])


    def __init__(self):
        self.print_resume()
        for _ in self.loop():
            self.worker()

    def worker(self):
        ...

    def loop(self):
        yield


class Base:
    operations: dict[str, Union[GlobalPaths, Callable]]

    def __init__(self, conf: str, role: Enum):
        self.operations[role.name].config = Config(Path(conf))
        self.operations[role.name]()
        print(f'\n====== The end of {role.name} ======')
