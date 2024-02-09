import datetime
import json
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage

from .util import splitx, run_command


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Config:
    config_data: dict

    def __init__(self, config_file: Union[Path, str]):
        print(f'Loading {config_file}.')

        self.config_file = Path(config_file)
        self.config_data = json.loads(self.config_file.read_text())

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
        return list(range(int(self.duration * int(self.fps))))

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
    def face_resolution(self) -> str:
        h, w, _ = self.face_shape
        return f'{w}x{h}'

    @property
    def face_shape(self) -> (int, int, int):
        h, w, _ = self.video_shape
        return round(h / 2), round(w / 3), 3

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.videos_list[self.video]['scale'])
        return h, w, 3

    @property
    def fps(self) -> str:
        return self.config['fps']

    @property
    def rate_control(self) -> str:
        return self.config['rate_control']

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


class GlobalPaths(Factors):
    worker_name: str = None
    overwrite = False
    dectime_folder = Path('dectime')
    graphs_folder = Path('graphs')
    operation_folder = Path('')

    @property
    def project_path(self) -> Path:
        return Path('results') / self.config['project']

    def tile_position(self):
        """
        Need video, tiling and tile
        :return: x1, x2, y1, y2
        """
        proj_h, proj_w = self.video_shape[:2]
        tiling_w, tiling_h = splitx(self.tiling)
        tile_w, tile_h = int(proj_w / tiling_w), int(proj_h / tiling_h)
        tile_m, tile_n = int(self.tile) % tiling_w, int(self.tile) // tiling_w
        x1 = tile_m * tile_w
        y1 = tile_n * tile_h
        x2 = tile_m * tile_w + tile_w  # not inclusive [...)
        y2 = tile_n * tile_h + tile_h  # not inclusive [...)
        return x1, y1, x2, y2


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


class Log:
    log_text: defaultdict
    video: str
    tiling: str
    quality: str
    tile: str
    chunk: str

    @contextmanager
    def logger(self):
        try:
            yield
        finally:
            self.save_log()

    def start_log(self):
        self.log_text = defaultdict(list)

    def log(self, error_code: str, filepath):
        self.log_text['video'].append(f'{self.video}')
        self.log_text['tiling'].append(f'{self.tiling}')
        self.log_text['quality'].append(f'{self.quality}')
        self.log_text['tile'].append(f'{self.tile}')
        self.log_text['chunk'].append(f'{self.chunk}')
        self.log_text['error'].append(error_code)
        self.log_text['parent'].append(f'{filepath.parent}')
        self.log_text['path'].append(f'{filepath.absolute()}')

    def save_log(self):
        cls_name = self.__class__.__name__
        filename = f'log_{cls_name}_{datetime.datetime.now()}.csv'
        filename = filename.replace(':', '-')
        df_log_text = pd.DataFrame(self.log_text)
        df_log_text.to_csv(filename, encoding='utf-8')


class Utils:
    command_pool: list
    config: Config
    project_path: Path
    segment_file: Path
    vid_proj: str
    name: str
    tiling: str
    quality: str
    tile: str
    chunk: str

    def print_state(self):
        print(f'Dectime [{self.vid_proj}][{self.name}][{self.tiling}][crf{self.quality}][tile{self.tile}]'
              f'[chunk{self.chunk}] = ', end='\r')

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

    @contextmanager
    def multi(self):
        self.command_pool = []
        try:
            yield
            with Pool(5) as p:
                p.map(run_command, self.command_pool)
            # for command in self.command_pool:
            #     self.run_command(command)
        finally:
            pass


class SiTi:
    filename: Path
    previous_frame: Optional[np.ndarray]
    siti: dict

    def __init__(self, filename: Path, verbose=True):
        self.filename = filename
        self.calc_siti(verbose=verbose)

    def calc_siti(self, verbose=True):
        vreader = skvideo.io.vreader(fname=str(self.filename), as_grey=True)
        si = []
        ti = []
        self.previous_frame = None
        name = f'{self.filename.parts[-2]}/{self.filename.name}'

        for frame_counter, frame in enumerate(vreader):
            # Fix shape
            frame = frame[0, :, :, 0].astype(np.float)

            value_si = self._calc_si(frame)
            si.append(value_si)
            try:
                value_ti = self._calc_ti(frame)
                ti.append(value_ti)
                if verbose:
                    print(f'\rCalculating SiTi - {name}: Frame {frame_counter}, si={value_si:.2f}, ti={value_ti:.3f}', end='')
            except TypeError:
                pass

        print('')
        self.siti = {'si': si,
                     'si_avg': np.average(si),
                     'si_std': np.std(si),
                     'si_max': np.max(si),
                     'si_min': np.min(si),
                     'si_med': np.median(si),
                     'ti': ti,
                     'ti_avg': np.average(ti),
                     'ti_std': np.std(ti),
                     'ti_max': np.max(ti),
                     'ti_min': np.min(ti),
                     'ti_med': np.median(ti)
                     }

    @staticmethod
    def _calc_si(frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Spatial Information for a video frame. Calculate both vectors and so the magnitude.
        :param frame: A luma video frame in numpy ndarray format.
        :return: spatial information and sobel frame.
        """
        sob_y = ndimage.sobel(frame, axis=0)
        sob_x = ndimage.sobel(frame, axis=1, mode="wrap")
        sobel = np.hypot(sob_y, sob_x)
        si = sobel.std()
        return si

    def _calc_ti(self, frame: np.ndarray) -> (float, np.ndarray):
        """
        Calcule Temporal Information for a video frame. If is a first frame,
        the information is zero.
        :param frame: A luma video frame in numpy ndarray format.
        :return: Temporal information and diference frame. If first frame the
        diference is zero array on same shape of frame.
        """
        try:
            difference = frame - self.previous_frame
            return difference.std()
        except TypeError:
            raise TypeError('First Frame')
        finally:
            self.previous_frame = frame
