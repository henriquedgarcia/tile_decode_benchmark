from __future__ import print_function
from enum import Enum

import datetime
import json
import time
from abc import ABC, abstractmethod
from builtins import PermissionError, object
from collections import Counter, defaultdict
from logging import warning, debug, fatal
from pathlib import Path
from subprocess import run, DEVNULL, STDOUT, PIPE
from typing import Any, Union, Dict, Optional, Generator, overload

import matplotlib.axes
import scipy.stats
import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import skvideo.io
from PIL import Image
from cycler import cycler
from fitter import Fitter

from .assets import AutoDict, Role
from .transform import cart2hcs
# from .util import (run_command, check_video_gop, iter_frame, load_sph_file,
from .util import (check_video_gop, iter_frame, load_sph_file,
                   save_json, load_json, lin_interpol, save_pickle,
                   load_pickle, splitx, idx2xy)
from .video_state import Config, VideoContext
from .viewport import ERP
from .nfov import NFOV
from itertools import combinations


def run_command(command: str, log_file: Path = None):
    print(command)
    process = run(command, shell=True, stderr=STDOUT, encoding='utf-8')
    if process.returncode != 0: warning(f'SUBPROCESS ERROR: Return {process.returncode} to video {log_file}. Continuing.')
    log_file.write_text(command + '\n')
    log_file.write_text(process.stdout + '\n')


class Factors:
    config: Config = None

    video: str = None
    quality_ref: str = '0'
    quality: str = None
    tiling: str = None
    metric: str = None
    tile: str = None
    chunk: str = None
    proj: str = None

    # Video context
    @property
    def name(self) -> str:
        name = self.video.replace('_cmp', '').replace('_erp', '')
        return name

    @property
    def name_list(self) -> list[str]:
        return list(set([video.replace('_cmp', '').replace('_erp', '') for video in self.videos_list]))

    @property
    def proj_list(self) -> list[str]:
        projs = set([self.videos_list[video]['projection'] for video in self.videos_list])
        return list(projs)

    @property
    def videos_list(self) -> Dict[str, Dict[str, Union[int, float, str]]]:
        return self.config.videos_list

    @property
    def tiling_list(self) -> list[str]:
        return self.config['tiling_list']

    @property
    def metric_list(self) -> list[str]:
        return ['time', 'rate', 'time_std', 'PSNR', 'WS-PSNR', 'S-PSNR']

    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        if '0' in quality_list: quality_list.remove('0')
        return quality_list

    @property
    def tile_list(self) -> list[str]:
        tile_m, tile_n = map(int, self.tiling.split('x'))
        return list(map(str, range(tile_n * tile_m)))

    @property
    def chunk_list(self) -> list[str]:
        return list(map(str, range(1, int(self.chunk_dur) + 1)))

    @property
    def offset(self) -> int:
        return int(self.videos_list[self.video]['offset'])

    @property
    def chunk_dur(self) -> int:
        return int(self.videos_list[self.video]['duration'])

    @property
    def resolution(self) -> str:
        return self.videos_list[self.video]['scale']

    @property
    def video_shape(self) -> tuple:
        w, h = splitx(self.videos_list[self.video]['scale'])
        return h, w, 3

    @property
    def original(self) -> str:
        return self.videos_list[self.video]['original']

    @property
    def vid_proj(self) -> str:
        return self.videos_list[self.video]['projection']

    @property
    def duration(self) -> str:
        return self.videos_list[self.video]['duration']

    @property
    def fps(self) -> str:
        return self.config['fps']

    @property
    def gop(self) -> str:
        return self.config['gop']

    @property
    def decoding_num(self) -> int:
        return int(self.config['decoding_num'])


class GlobalPaths(Factors):
    dectime_folder = Path('dectime')
    get_tiles_folder = Path('get_tiles')
    quality_folder = Path('quality')
    segment_folder = Path('segment')
    graphs_folder = Path('graphs')

    @property
    def project_path(self) -> Path:
        return Path('results') / self.config['project']

    @property
    def workfolder(self) -> Path:
        folder = self.project_path / self.graphs_folder / f'{self.__class__.__name__}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder_data(self) -> Path:
        folder = self.workfolder / 'data'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def dectime_result_json(self) -> Path:
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'times_{self.video}.json'

    @property
    def bitrate_result_json(self) -> Path:
        folder = self.project_path / self.segment_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'rate_{self.video}.json'

    @property
    def get_tiles_result_json(self) -> Path:
        folder = self.project_path / self.get_tiles_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'get_tiles_{self.config["dataset_name"]}_{self.video}_{self.tiling}.json'

    @property
    def quality_result_json(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'quality_{self.video}.json'


# class BaseTileDecodeBenchmark:
#     config: Config = None
#     video_context: VideoContext = None
#     role: Role = None
#
#     def run(self, **kwargs):
#         self.print_resume()
#         self.role.init()
#
#         for n in self.video_context:
#             action = self.role.operation(**kwargs)
#
#             if action in (None, 'continue', 'skip'):
#                 continue
#             elif action in ('break',):
#                 break
#             elif action in ('exit',):
#                 return
#
#         self.role.finish()
#         print(f'The end of {self.role.name}')
#
#     def print_resume(self):
#         print('=' * 70)
#         print(f'Processing {len(self.config.videos_list)} videos:\n'
#               f'  operation: {self.role.name}\n'
#               f'  project: {self.video_context.project_path}\n'
#               f'  codec: {self.config["codec"]}\n'
#               f'  fps: {self.config["fps"]}\n'
#               f'  gop: {self.config["gop"]}\n'
#               f'  qualities: {self.config["quality_list"]}\n'
#               f'  patterns: {self.config["tiling_list"]}'
#               )
#         print('=' * 70)
#
#     def count_decoding(self) -> int:
#         """
#         Count how many times the word "utime" appears in "log_file"
#         :return:
#         """
#         dectime_log = self.video_context.dectime_log
#         try:
#             content = dectime_log.read_text(encoding='utf-8').splitlines()
#         except UnicodeDecodeError:
#             warning('ERROR: UnicodeDecodeError. Cleaning.')
#             dectime_log.unlink()
#             return 0
#         except FileNotFoundError:
#             warning('ERROR: FileNotFoundError. Return 0.')
#             return 0
#
#         return len(['' for line in content if 'utime' in line])
#
#     def get_times(self) -> List[float]:
#         times = []
#         dectime_log = self.video_context.dectime_log
#
#         for line in dectime_log.read_text(encoding='utf-8').splitlines():
#             dectime = line.strip().split(' ')[1].split('=')[1][:-1]
#             times.append(float(dectime))
#
#         content = self.video_context.dectime_log.read_text(encoding='utf-8')
#         content_lines = content.splitlines()
#         times = [float(line.strip().split(' ')[1].split('=')[1][:-1])
#                  for line in content_lines if 'utime' in line]
#         return times


class Worker(ABC, GlobalPaths):
    fit: Fitter
    fitter = AutoDict()
    fit_errors = AutoDict()
    data: Union[dict, AutoDict] = None
    stats: dict[str, list] = None
    seen_tiles_data = None
    graphs_folder: Path
    color_list = {'burr12': 'tab:blue',
                  'fatiguelife': 'tab:orange',
                  'gamma': 'tab:green',
                  'invgauss': 'tab:red',
                  'rayleigh': 'tab:purple',
                  'lognorm': 'tab:brown',
                  'genpareto': 'tab:pink',
                  'pareto': 'tab:gray',
                  'halfnorm': 'tab:olive',
                  'expon': 'tab:cyan'}
    dists_colors = {'burr12': 'tab:blue',
                    'fatiguelife': 'tab:orange',
                    'gamma': 'tab:green',
                    'invgauss': 'tab:red',
                    'rayleigh': 'tab:purple',
                    'lognorm': 'tab:brown',
                    'genpareto': 'tab:pink',
                    'pareto': 'tab:gray',
                    'halfnorm': 'tab:olive',
                    'expon': 'tab:cyan'}

    def video_context(self) -> Generator: ...

    def work(self): ...

    @abstractmethod
    def __init__(self, config):
        self.config = config
        self.print_resume()

        for _ in self.video_context():
            self.work()

    @staticmethod
    def count_decoding(dectime_log: Path) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :return:
        """
        try:
            content = dectime_log.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            warning('ERROR: UnicodeDecodeError. Cleaning.')
            dectime_log.unlink()
            return 0
        except FileNotFoundError:
            warning('ERROR: FileNotFoundError. Return 0.')
            return 0

        return len(['' for line in content if 'utime' in line])

    @staticmethod
    def run_command(command: str, log_file: Path = None, mode: str = None):
        """
        Run a shell command with subprocess module with realtime output.
        :param command: A command string to run.
        :param log_file: A path-like to save the process output.
        :param mode: The write mode: 'w' or 'a'.
        :return: stdout.
        """
        # Run
        process = run(command, shell=True, stdout=PIPE, stderr=STDOUT, encoding='utf-8', )
        if process.returncode != 0: warning(f'SUBPROCESS ERROR: Return {process.returncode} to video {log_file}. Continuing.')

        # Write in Logfile
        # Check logfile
        try:
            log_file.unlink(missing_ok=True)
        except PermissionError:
            print(f'run_command() can\'t delete old logfile {log_file} (PermissionError).')
            return

        log_file.write_text(command + '\n')
        log_file.write_text(process.stdout + '\n')

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


class ____Benchmark____: ...

class TileDecodeBenchmarkPaths(GlobalPaths):
        # Folders
        original_folder = Path('original')
        lossless_folder = Path('lossless')
        compressed_folder = Path('compressed')
        _viewport_folder = Path('viewport')
        _siti_folder = Path('siti')
        _check_folder = Path('check')

        @property
        def basename(self):
            return Path(f'{self.name}_'
                        f'{self.resolution}_'
                        f'{self.config["fps"]}_'
                        f'{self.tiling}_'
                        f'{self.config["rate_control"]}{self.quality}')

        @property
        def original_file(self) -> Path:
            return self.original_folder / self.original

        @property
        def lossless_file(self) -> Path:
            folder = self.project_path / self.lossless_folder
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.video}_{self.resolution}_{self.config["fps"]}.mp4'

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
        def reference_segment(self) -> Union[Path, None]:
            # 'segment/angel_falls_nas_4320x2160_30_12x8_crf0/tile11_030.mp4'
            basename = Path(f'{self.name}_'
                            f'{self.resolution}_'
                            f'{self.fps}_'
                            f'{self.tiling}_'
                            f'{self.config["rate_control"]}{self.config["original_quality"]}')
            folder = self.project_path / self.segment_folder / basename
            return folder / f'tile{self.tile}_{int(self.chunk):03d}.mp4'

        @property
        def dectime_log(self) -> Path:
            folder = self.project_path / self.dectime_folder / self.basename
            folder.mkdir(parents=True, exist_ok=True)
            chunk = int(str(self.chunk))
            return folder / f'tile{self.tile}_{chunk:03d}.log'

class TileDecodeBenchmark:
    config: Config = None
    results = AutoDict()
    results_dataframe = pd.DataFrame()
    video_context: VideoContext = None
    role: Role = None


    def __init__(self, config: str = None, role: str = None):
        operations = {'PREPARE': self.Prepare, 'COMPRESS': self.Compress,
                      'SEGMENT': self.Prepare, 'DECODE': self.Prepare,
                      'COLLECT_RESULTS': self.Prepare}

        operations[role](Config(config))
        print(f'\n====== The end of {role} ======')

    # PREPARE
    class Prepare(Worker, TileDecodeBenchmarkPaths):
        def video_context(self):
            for self.video in self.videos_list:
                yield

        def __init__(self, config:Config):
            self.config = config
            self.print_resume()
            for _ in self.video_context():
                self.work(overwrite=False)

        def work(self, overwrite=False):
            original_file: Path = self.original_file
            lossless_file: Path = self.lossless_file
            lossless_log: Path = self.lossless_file.with_suffix('.log')
            print(f'Preparing {self.lossless_file=}', end='')

            if lossless_file and not overwrite:
                warning(f'  The file {lossless_file=} exist. Skipping.')
                return

            if not original_file.exists():
                warning(f'  The file {original_file=} not exist. Skipping.')
                return

            resolution_ = splitx(self.resolution)
            dar = resolution_[0] / resolution_[1]

            cmd = f'ffmpeg '
            cmd += f'-hide_banner -y '
            cmd += f'-ss {self.offset} '
            cmd += f'-i {original_file} '
            cmd += f'-crf 0 '
            cmd += f'-t {self.duration} '
            cmd += f'-r {self.fps} '
            cmd += f'-map 0:v '
            cmd += f'-vf "scale={self.resolution},setdar={dar}" '
            cmd += f'{lossless_file}'

            print(cmd)
            run_command(cmd, lossless_log, 'w')

    # COMPRESS
    class Compress(Worker, TileDecodeBenchmarkPaths):
        @property
        def quality_list(self):
            quality_list = self.config['quality_list']
            return quality_list

        def video_context(self):
            for self.video in self.videos_list:
                for self. tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            yield

        def __init__(self, config: Config):
            self.config = config
            self.print_resume()
            self.work(overwrite=False)

        def work(self, overwrite=False):
            commands_list = []
            for self.video in self.videos_list:
                for self. tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            print(f'\r{self.compressed_file}', end='')
                            if self.compressed_file.exists() and not overwrite:
                                warning(f'The file {self.compressed_file} exist. Skipping.')
                                return

                            if not self.lossless_file.exists():
                                warning(f'The file {self.lossless_file} not exist. Skipping.')
                                return

                            pw, ph = splitx(self.resolution)
                            M, N = splitx(self.tiling)
                            tw, th = int(pw / M), int(ph / N)
                            tx, ty = int(self.tile) * tw, int(self.tile) * th
                            factor = self.config["rate_control"]

                            cmd = ['bin/ffmpeg -hide_banner -y -psnr']
                            cmd += [f'-i {self.lossless_file}']
                            cmd += [f'-c:v libx265']
                            cmd += [f'-{factor} {self.quality} -tune "psnr"']
                            cmd += [f'-x265-params']
                            cmd += [f'"keyint={self.gop}:'
                                    f'min-keyint={self.gop}:'
                                    f'open-gop=0:'
                                    f'scenecut=0:'
                                    f'info=0"']
                            cmd += [f'-vf "crop='
                                    f'w={tw}:h={th}:'
                                    f'x={tx}:y={ty}"']
                            cmd += [f'{self.compressed_file}']
                            cmd = ' '.join(cmd)
                            compressed_log = self.compressed_file.with_suffix('.log')
                            self.run_command(cmd, compressed_log, 'w')

    # SEGMENT
    class Segment(Worker, TileDecodeBenchmarkPaths):
        @property
        def quality_list(self):
            quality_list = self.config['quality_list']
            return quality_list

        def video_context(self):
            for self.video in self.videos_list:
                for self. tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.tile in self.tiling_list:
                            yield

        def work(self, overwrite=False) -> Any:
            segment_log = self.segment_file.with_suffix('.log')

            print(f'==== Processing {self.segment_folder} ====')

            # If segment log size is very small, infers error and overwrite.
            if segment_log.is_file() and segment_log.stat().st_size > 10000 and not overwrite:
                warning(f'The file {segment_log} exist. Skipping.')
                return

            if not self.compressed_file.is_file():
                warning(f'The file {self.compressed_file} not exist. Skipping.')
                return

            # todo: Alternative:
            # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
            # ime 1 -reset_timestamps 1 output%03d.mp4

            cmd = ['MP4Box']
            cmd += ['-split 1']
            cmd += [f'{self.compressed_file}']
            cmd += [f'-out {self.segment_folder}/']
            cmd = ' '.join(cmd)
            cmd = f'bash -c "{cmd}"'

            self.run_command(cmd, segment_log, 'w')

    # DECODE
    class Decode(Worker, TileDecodeBenchmarkPaths):
        def video_context(self):
            for self.video in self.videos_list:
                for self. tiling in self.tiling_list:
                    for self.quality in self.quality_list:

                        for i in range(self.decoding_num):
                            for self.tile in self.tiling_list:
                                for self.chunk in self.chunk_list:
                                    yield

        def work(self, overwrite=False) -> Any:
            print(f'Decoding {self.segment_file=}', end = '')

            if self.dectime_log.exists():
                if self.decoding_num - self.count_decoding(self.dectime_log) <= 0 and not overwrite:
                    warning(f'  {self.segment_file} is decoded enough. Skipping.')
                    return

            if not self.segment_file.is_file():
                warning(f'  The file {self.segment_file} not exist. Skipping.')
                return

            cmd = (f'ffmpeg -hide_banner -benchmark '
                   f'-codec hevc -threads 1 '
                   f'-i {self.segment_file} '
                   f'-f null -')

            run_command(cmd,  self.dectime_log, 'a')

    # COLLECT RESULTS
    class CollectDectime(Worker, TileDecodeBenchmarkPaths):
        """
        The result dict have a following structure:
        results[video_name][tile_pattern][quality][idx][chunk_id]
                ['utime'|'bit rate']['psnr'|'qp_avg']
        [video_name]    : The video name
        [tile_pattern]  : The tile tiling. e.g. "6x4"
        [quality]       : Quality. An int like in crf or qp.
        [idx]           : the tile number. ex. max = 6*4
        if [chunk_id]   : A id for chunk. With 1s chunk, 60s video have 60
                          chunks
            [type]      : "utime" (User time), or "bit rate" (Bit rate in kbps)
                          of a chunk.
        if ['psnr']     : the ffmpeg calculated psnr for tile (before
                          segmentation)
        if ['qp_avg']   : The ffmpeg calculated average QP for an encoding.
        """
        result_times = AutoDict()
        result_rate = AutoDict()

        def video_context(self):
            for self.video in self.videos_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.tile in self.tiling_list:
                            for self.chunk in self.chunk_list:
                                yield
            self.save_dectime()

        def work(self, overwrite=False) -> Any:
            print(f'Collecting {self.video_context}')

            ## Collect Bitrate
            if self.dectime_result_json.exists() and not overwrite:
                warning(f'The file {self.dectime_result_json} exist and not overwrite. Skipping.')
                return 'exit'
            result_rate:AutoDict = self.result_rate[self.vid_proj][self.video][self.tiling][self.quality][self.tile][self.chunk]
            if self.segment_file.exists():
                try:
                    chunk_size = self.segment_file.stat().st_size
                    bitrate = 8 * chunk_size / self.chunk_dur
                    result_rate['rate'] = bitrate
                except PermissionError:
                    warning(f'PermissionError error on reading size of {self.segment_file}.')
                    result_rate['rate'] = 0
            else:
                warning(f'The chunk {self.segment_file} not exist_ok.')
                result_rate['rate'] = 0

            if not self.dectime_log.exists():
                warning(f'The dectime log {self.dectime_log} not exist. Skipping.')
                return 'skip'

            ## Collect Dectime
            # times = []
            content = self.dectime_log.read_text(encoding='utf-8').splitlines()
            times = [float(line.strip().split(' ')[1].split('=')[1][:-1])
                     for line in content if 'utime' in line]

            data = {'times': times}
            print(f'\r{data}', end='')

            results_times = self.result_times[self.vid_proj][self.video][self.tiling][self.quality][self.tile][self.chunk]

            results_times.update(data)

        def save_dectime(self):
            filename = self.dectime_result_json
            save_json(self.result_times, filename)

            filename = self.bitrate_result_json
            save_json(self.result_rate, filename)


class ____Quality____: ...


# class QualityMetrics:
#     PIXEL_MAX: int = 255
#     video_context: VideoContext = None
#     weight_ndarray: Union[np.ndarray, object] = np.zeros(0)
#     sph_points_mask: np.ndarray = np.zeros(0)
#     sph_points_img: list = []
#     sph_points: list = []
#     cart_coord: list = []
#     sph_file: Path
#     results: AutoDict
#
#     # ### Coordinate system ### #
#     # Image coordinate system
#     ICSPoint = NamedTuple('ICSPoint', (('x', float), ('y', float)))
#     # Horizontal coordinate system
#     HCSPoint = NamedTuple('HCSPoint',
#                           (('azimuth', float), ('elevation', float)))
#     # Aerospacial coordinate system
#     ACSPoint = NamedTuple('ACSPoint',
#                           (('yaw', float), ('pitch', float), ('roll', float)))
#     # Cartesian coordinate system
#     CCSPoint = NamedTuple('CCSPoint',
#                           (('x', float), ('y', float), ('z', float)))
#
#     # ### util ### #
#     def mse2psnr(self, mse: float) -> float:
#         return 10 * np.log10((self.PIXEL_MAX ** 2 / mse))
#
#     # ### psnr ### #
#     def psnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
#              im_sal: np.ndarray = None) -> float:
#         """
#         https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
#
#         Images must be only one channel (luminance)
#         (height, width) = im_ref.shape()
#         "float32" = im_ref.dtype()
#
#         :param im_ref:
#         :param im_deg:
#         :param im_sal:
#         :return:
#         """
#         im_sqr_err = (im_ref - im_deg) ** 2
#         if im_sal is not None:
#             im_sqr_err = im_sqr_err * im_sal
#         mse = np.average((im_ref - im_deg) ** 2)
#         return self.mse2psnr(mse)
#
#         # # separate the channels to color image
#         # psnr_ = []
#         # if len(im_ref.shape[-1]) > 2:
#         #     for channel in range(im_ref.shape[-1]):
#         #         im_ref_ = im_ref[..., channel]
#         #         im_deg_ = im_deg[..., channel]
#         #         psnr_.append(psnr(im_ref_, im_deg_, im_sal))
#         # else:
#         #     psnr_.append(psnr(im_ref, im_deg, im_sal))
#         #
#         #
#         #     # for channel in
#         #     pass
#         #
#         # return
#
#     # ### wspsnr ### #
#     def prepare_weight_ndarray(self):
#         if self.video_context.video.projection == 'erp':
#             height, width = self.video_context.video.resolution.shape
#             func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi
#                                        / height)
#             self.weight_ndarray = np.fromfunction(func, (height, width),
#                                                   dtype='float32')
#         elif self.video_context.video.projection == 'cmp':
#             # each face must be square (proj. aspect ration == 3:2).
#             face = self.video_context.video.resolution.shape[0] / 2
#             face_ctr = face / 2
#
#             squared_dist = lambda y, x: (
#                     (x + 0.5 - face_ctr) ** 2 + (y + 0.5 - face_ctr) ** 2
#             )
#
#             func = lambda y, x: (
#                     (1 + squared_dist(y, x) / (face_ctr ** 2)) ** (-3 / 2)
#
#             )
#
#             weighted_face = np.fromfunction(func, (int(face), int(face)),
#                                             dtype='float32')
#             weight_ndarray = np.concatenate((weighted_face,
#                                              weighted_face))
#             self.weight_ndarray = np.concatenate(
#                 (weight_ndarray, weight_ndarray, weight_ndarray),
#                 axis=1)
#         else:
#             fatal(f'projection {self.video_context.video.projection} not supported.')
#             raise FileNotFoundError(
#                 f'projection {self.video_context.video.projection} not supported.')
#
#     def wspsnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
#                im_sal: np.ndarray = None):
#         """
#         Must be same size
#         :param im_ref:
#         :param im_deg:
#         :param im_sal:
#         :return:
#         """
#
#         if self.video_context.video.resolution.shape != self.weight_ndarray.shape:
#             self.prepare_weight_ndarray()
#
#         x1 = self.video_context.tile.position.x
#         x2 = self.video_context.tile.position.x + self.video_context.tile.resolution.W
#         y1 = self.video_context.tile.position.y
#         y2 = self.video_context.tile.position.y + self.video_context.tile.resolution.H
#         weight_tile = self.weight_ndarray[y1:y2, x1:x2]
#
#         tile_weighted = weight_tile * (im_ref - im_deg) ** 2
#
#         if im_sal is not None:
#             tile_weighted = tile_weighted * im_sal
#         wmse = np.average(tile_weighted)
#
#         if wmse == 0:
#             return 1000
#
#         return self.mse2psnr(wmse)
#
#     # ### spsnr_nn ### #
#     def spsnr_nn(self, im_ref: np.ndarray,
#                  im_deg: np.ndarray,
#                  im_sal: np.ndarray = None):
#         """
#         Calculate of S-PSNR between two images. All arrays must be on the same
#         resolution.
#
#         :param im_ref: The original image
#         :param im_deg: The image degraded
#         :param im_sal: The saliency map
#         :return:
#         """
#         # sph_file = Path('lib/sphere_655362.txt'),
#         shape = self.video_context.video.resolution.shape
#
#         if self.sph_points_mask.shape != shape:
#             sph_file = load_sph_file(self.sph_file, shape)
#             self.sph_points_mask = sph_file[-1]
#
#         x1 = self.video_context.tile.position.x
#         x2 = self.video_context.tile.position.x + self.video_context.tile.resolution.W
#         y1 = self.video_context.tile.position.y
#         y2 = self.video_context.tile.position.y + self.video_context.tile.resolution.H
#         mask = self.sph_points_mask[y1:y2, x1:x2]
#
#         im_ref_m = im_ref * mask
#         im_deg_m = im_deg * mask
#
#         sqr_dif: np.ndarray = (im_ref_m - im_deg_m) ** 2
#
#         if im_sal is not None:
#             sqr_dif = sqr_dif * im_sal
#
#         mse = sqr_dif.sum() / mask.sum()
#         return self.mse2psnr(mse)
#
#     def ffmpeg_psnr(self):
#         if self.video_context.chunk == 1:
#             name, pattern, quality, tile, chunk = self.video_context.state
#             results = self.results[name][pattern][str(quality)][tile]
#             results.update(self._collect_ffmpeg_psnr())
#
#     def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
#         get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
#         get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
#         psnr = None
#         compressed_log = self.video_context.compressed_file.with_suffix('.log')
#         content = compressed_log.read_text(encoding='utf-8')
#         content = content.splitlines()
#
#         for line in content:
#             if 'Global PSNR' in line:
#                 psnr = {'psnr': get_psnr(line),
#                         'qp_avg': get_qp(line)}
#                 break
#         return psnr

class QualityAssessmentPaths(TileDecodeBenchmarkPaths):
    @property
    def quality_video_csv(self) -> Union[Path, None]:
        folder = self.project_path / self.quality_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.mp4.csv'


class QualityAssessment:
    def __init__(self, config: str, role: str):
        operations = {'ALL_METRICS': self.AllMetrics,
                      'COLLECT_RESULTS': self.ColectResults,
                      }

        operations[role](Config(config))
        print(f'\n====== The end of {role} ======')

    class AllMetrics(Worker, QualityAssessmentPaths):
        video_context: np.ndarray

        def video_context(self):
            for self.video in self.videos_list:
                for self. tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.tile in self.tiling_list:
                            for self.chunk in self.chunk_list:
                                yield

        def __init__(self, config: str):
            """
            Load configuration and run the main routine defined on Role Operation.

            :param config: a Config object
            """
            self.config = Config(config)
            self.print_resume()

            self.metrics = {'MSE': self._mse,
                            'WS-MSE': self._wsmse,
                            'S-MSE': self._smse_nn}

            self.sph_points_mask: np.ndarray = np.zeros(0)
            self.old_video = None
            self.chunk_quality = defaultdict(list)
            self.results = AutoDict()
            self.sph_file = Path('lib/sphere_655362.txt')
            self.results_dataframe = pd.DataFrame()
            self.old_video = ''

            for _ in self.video_context():
                self.all(overwrite=False)

        def all(self, overwrite=False):
            if self.old_video is not None \
                    and self.old_video != self.video\
                    and self.old_video != self.video\
                    and self.old_video != self.video\
                    :
                self.old_video = self.video
                pd.DataFrame(self.chunk_quality).to_csv(self.quality_video_csv, encoding='utf-8', index_label='frame')

            debug(f'Processing [{self.proj}][{self.video}][{self.tiling}][{self.tile}][{self.chunk}]')
            if self.quality_video_csv.exists() and not overwrite:
                warning(f'The chunk quality csv {self.quality_video_csv} exist. Skipping')
                return

            reference_file = self.reference_file
            segment_file = self.segment_file
            if not (segment_file.exists() and reference_file.exists()):
                warning(f'Some file not exist. Skipping')
                return

            csv_dataframe = pd.DataFrame()
            chunk_quality = defaultdict(list)

            frames = zip(iter_frame(reference_file), iter_frame(segment_file))
            start = time.time()

            for n, (frame_video1, frame_video2) in enumerate(frames):
                for metric in self.metrics:
                    if metric in csv_dataframe: continue
                    metrics_method = self.metrics[metric]
                    metric_value = metrics_method(frame_video1, frame_video2)
                    chunk_quality[metric].append(metric_value)
                    psnr = self._mse2psnr(metric_value)
                    chunk_quality[metric.replace('MSE', 'PSNR')].append(psnr)

                print(f'[{self.proj}][{self.video}][{self.tiling}][{self.tile}][{self.chunk}] - '
                      f'Frame {n} - {time.time() - start: 0.3f} s', end='\r')

            print('')

        # ### psnr ### #
        def _mse(self, im_ref: np.ndarray, im_deg: np.ndarray, im_sal: np.ndarray = None) -> float:
            """
            https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

            Images must be only one channel (luminance)
            (height, width) = im_ref.shape()
            "float32" = im_ref.dtype()

            :param im_ref:
            :param im_deg:
            :param im_sal:
            :return:
            """
            im_sqr_err = (im_ref - im_deg) ** 2
            if im_sal is not None:
                im_sqr_err = im_sqr_err * im_sal
            mse = np.average(im_sqr_err)
            return self._mse2psnr(mse)

            # # separate the channels to color image
            # psnr_ = []
            # if len(im_ref.shape[-1]) > 2:
            #     for channel in range(im_ref.shape[-1]):
            #         im_ref_ = im_ref[..., channel]
            #         im_deg_ = im_deg[..., channel]
            #         psnr_.append(psnr(im_ref_, im_deg_, im_sal))
            # else:
            #     psnr_.append(psnr(im_ref, im_deg, im_sal))
            #
            #
            #     # for channel in
            #     pass
            #
            # return

        # ### wspsnr ### #
        def _wsmse(self, im_ref: np.ndarray, im_deg: np.ndarray, im_sal: np.ndarray = None) -> float:
            """
            Must be same size
            :param im_ref:
            :param im_deg:
            :param im_sal:
            :return:
            """

            if self.video_context.video.resolution.shape != self.weight_ndarray.shape:
                self._prepare_weight_ndarray()

            x1 = self.video_context.tile.position.x
            x2 = self.video_context.tile.position.x + self.video_context.tile.resolution.W
            y1 = self.video_context.tile.position.y
            y2 = self.video_context.tile.position.y + self.video_context.tile.resolution.H
            weight_tile = self.weight_ndarray[y1:y2, x1:x2]

            tile_weighted = weight_tile * (im_ref - im_deg) ** 2

            if im_sal is not None:
                tile_weighted = tile_weighted * im_sal
            wmse = np.average(tile_weighted)

            if wmse == 0:
                return 1000

            return wmse

        # ### spsnr_nn ### #
        def _smse_nn(self, im_ref: np.ndarray, im_deg: np.ndarray, im_sal: np.ndarray = None):
            """
            Calculate of S-PSNR between two images. All arrays must be on the same
            resolution.

            :param im_ref: The original image
            :param im_deg: The image degraded
            :param im_sal: The saliency map
            :return:
            """
            shape = self.video_context.video.resolution.shape

            if self.sph_points_mask.shape != shape:
                sph_file = load_sph_file(self.sph_file, shape)
                self.sph_points_mask = sph_file[-1]

            x1 = self.video_context.tile.position.x
            x2 = self.video_context.tile.position.x + self.video_context.tile.resolution.W
            y1 = self.video_context.tile.position.y
            y2 = self.video_context.tile.position.y + self.video_context.tile.resolution.H
            mask = self.sph_points_mask[y1:y2, x1:x2]

            im_ref_m = im_ref * mask
            im_deg_m = im_deg * mask

            sqr_dif: np.ndarray = (im_ref_m - im_deg_m) ** 2

            if im_sal is not None:
                sqr_dif = sqr_dif * im_sal

            smse_nn = sqr_dif.sum() / mask.sum()
            return smse_nn

        # ### Util ### #
        def _mse2psnr(self, mse: float) -> float:
            return 10 * np.log10((255. ** 2 / mse))

        def _prepare_weight_ndarray(self):
            if self.video_context.video.projection == 'erp':
                height, width = self.video_context.video.resolution.shape
                func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi
                                           / height)
                self.weight_ndarray = np.fromfunction(func, (height, width),
                                                      dtype='float32')
            elif self.video_context.video.projection == 'cmp':
                # each face must be square (proj. aspect ration == 3:2).
                face = self.video_context.video.resolution.shape[0] / 2
                face_ctr = face / 2

                squared_dist = lambda y, x: (
                        (x + 0.5 - face_ctr) ** 2 + (y + 0.5 - face_ctr) ** 2
                )

                func = lambda y, x: (
                        (1 + squared_dist(y, x) / (face_ctr ** 2)) ** (-3 / 2)

                )

                weighted_face = np.fromfunction(func, (int(face), int(face)),
                                                dtype='float32')
                weight_ndarray = np.concatenate((weighted_face,
                                                 weighted_face))
                self.weight_ndarray = np.concatenate(
                    (weight_ndarray, weight_ndarray, weight_ndarray),
                    axis=1)
            else:
                fatal(f'projection {self.video_context.video.projection} not supported.')
                raise FileNotFoundError(
                    f'projection {self.video_context.video.projection} not supported.')


    class ColectResults: pass
        # def init_result(self):
        #     quality_result_json = self.video_context.quality_result_json
        #
        #     if quality_result_json.exists():
        #         warning(f'The file {quality_result_json} exist. Loading.')
        #         json_content = quality_result_json.read_text(encoding='utf-8')
        #         self.results = load_json(json_content)
        #     else:
        #         self.results = AutoDict()
        #
        # def result(self, overwrite=False):
        #     debug(f'Processing {self.video_context}')
        #     if self.video_context.quality == self.video_context.original_quality:
        #         # info('Skipping original quality')
        #         return 'continue'
        #
        #     results = self.results
        #     quality_csv = self.video_context.quality_video_csv  # The compressed quality
        #
        #     if not quality_csv.exists():
        #         warning(f'The file {quality_csv} not exist. Skipping.')
        #         return 'continue'
        #
        #     csv_dataframe = pd.read_csv(quality_csv, encoding='utf-8', index_col=0)
        #
        #     for key in self.video_context.state:
        #         results = results[key]
        #
        #     for metric in self.metrics:
        #         if results[metric] != {} and not overwrite:
        #             warning(f'The metric {metric} exist for Result '
        #                     f'{self.video_context.state}. Skipping this metric')
        #             return
        #
        #         try:
        #             results[metric] = csv_dataframe[metric].tolist()
        #             if len(results[metric]) == 0:
        #                 raise KeyError
        #         except KeyError:
        #             warning(f'The metric {metric} not exist for csv_dataframe'
        #                     f'{self.video_context.state}. Skipping this metric')
        #             return
        #     if self.old_video != f'{self.video_context.video}':
        #         self.old_video = f'{self.video_context.video}'
        #         self.save_result()
        #
        #     return 'continue'
        #
        # def save_result(self):
        #     quality_result_json = self.video_context.quality_result_json
        #     save_json(self.results, quality_result_json)
        #
        #     quality_result_pickle = quality_result_json.with_suffix('.pickle')
        #     save_pickle(self.results, quality_result_pickle)
        #
        # def ffmpeg_psnr(self):
        #     if self.video_context.chunk == 1:
        #         name, pattern, quality, tile, chunk = self.video_context.state
        #         results = self.results[name][pattern][quality][tile]
        #         results.update(self._collect_ffmpeg_psnr())
        #
        # def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        #     get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        #     get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        #     psnr = None
        #     compressed_log = self.video_context.compressed_file.with_suffix('.log')
        #     content = compressed_log.read_text(encoding='utf-8')
        #     content = content.splitlines()
        #
        #     for line in content:
        #         if 'Global PSNR' in line:
        #             psnr = {'psnr': get_psnr(line),
        #                     'qp_avg': get_qp(line)}
        #             break
        #     return psnr

        # ### util ### #





class ____Graphs____: ...

class DectimeGraphsPaths(GlobalPaths):
    workfolder_data: Path
    workfolder: Path

    bins: Union[int, str]

    # Data Bucket
    @property
    def data_bucket_file(self) -> Path:
        path = self.workfolder_data / f'data_bucket.json'
        return path

    @property
    def seen_tiles_data_file(self) -> Path:
        path = self.project_path / self.graphs_folder / f'seen_tiles.json'
        return path

    # Stats file
    @property
    def stats_file(self) -> Path:
        stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
        return stats_file

    @property
    def correlations_file(self) -> Path:
        correlations_file = self.workfolder / f'correlations.csv'
        return correlations_file


class DectimeGraphs:
    def __init__(self, config: str, role: str):
        operations = {'BY_PATTERN': self.ByPattern,
                      'BY_PATTERN_BY_QUALITY': self.ByPatternByQuality,
                      'BY_PATTERN_FULL_FRAME': self.ByPattern,
                      'BY_VIDEO_BY_PATTERN_BY_QUALITY': self.ByPattern,
                      'BY_VIDEO_BY_PATTERN_BY_TILE_BY_CHUNK': self.ByVideoByPatternByQualityByTile,
                      'BY_VIDEO_BY_PATTERN_BY_TILE_BY_QUALITY_BY_CHUNK': self.ByVideoByPatternByTileByQualityByChunk,
                      }

        operations[role](Config(config))
        print(f'\n====== The end of {role} ======')

    class ByPattern(Worker, DectimeGraphsPaths):
        dataset_name = 'nasrabadi_28videos'

        # Fitter file
        @property
        def fitter_pickle_file(self) -> Path:
            fitter_file = self.workfolder_data / f'fitter_{self.metric}_{self.proj}_{self.tiling}_{self.bins}bins.pickle'
            return fitter_file

        # data_bucket: AutoDict

        # def video_context(self):
        #     lv1 = []
        #     lv2 = []
        #     lv3 = []
        #     lv4 = []
        #     lv5 = []
        #     for self.video in self.videos_list:
        #         for self. tiling in self.tiling_list:
        #             for self.quality in self.quality_list:
        #                 for self.tile in self.tile_list:
        #                     for self.chunk in self.chunk_list:
        #                         lv1.append(self.video)
        #                         lv2.append(self.tiling)
        #                         lv3.append(self.quality)
        #                         lv4.append(self.tile)
        #                         lv5.append(self.chunk)
        #     index = pd.MultiIndex.from_tuples(zip(tuple(lv1), tuple(lv2), tuple(lv3), tuple(lv4), tuple(lv5)),
        #                                       names=["video", "tiling", 'quality', 'tile', 'chunk'])
        #     save_json([lv1, lv2, lv3, lv4, lv5],'index.json')
        #     yield

        def video_context(self):
            for self.metric in self.metric_list:
                for self.proj in ['erp']:
                    for self.tiling in self.tiling_list:
                        yield

        def __init__(self, config:Config):
            self.config = config
            self.print_resume()
            self.n_dist = 6
            self.bins = 30
            self.stats = defaultdict(list)
            self.corretations_bucket = defaultdict(list)

            self.get_data_bucket(overwrite=False)
            self.make_fit(overwrite=False)
            self.calc_stats(overwrite=False)

            DectimeGraphs.rc_config()
            self.make_hist(overwrite=True)
            self.make_boxplot(overwrite=False)
            self.make_violinplot(overwrite=False)

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            # Check file
            if not overwrite and self.data_bucket_file.exists():
                print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                return

            data_bucket = AutoDict()
            json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                           'time': self.dectime_result_json,
                                           'time_std': self.dectime_result_json,
                                           'PSNR': self.quality_result_json,
                                           'WS-PSNR': self.quality_result_json,
                                           'S-PSNR': self.quality_result_json}[metric]

            def bucket():
                # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
                # 1x1 - 10014 chunks - 1/181 tiles por tiling
                # 3x2 - 60084 chunks - 6/181 tiles por tiling
                # 6x4 - 240336 chunks - 24/181 tiles por tiling
                # 9x6 - 540756 chunks - 54/181 tiles por tiling
                # 12x8 - 961344 chunks - 96/181 tiles por tiling
                # total - 1812534 chunks - 181/181 tiles por tiling

                data = data_bucket[self.metric][self.vid_proj][self.tiling]
                if not isinstance(data, list):
                    data = data_bucket[self.metric][self.vid_proj][self.tiling] = []
                return data

            def process(value):
                # Process value according the metric
                if self.metric == 'time':
                    new_value = float(np.average(value['times']))
                elif self.metric == 'time_std':
                    new_value = float(np.std(value['times']))
                elif self.metric == 'rate':
                    new_value = float(value['rate'])
                else:
                    # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                    metric_value = value[self.metric]
                    new_value = metric_value
                    new_value = new_value if float(new_value) != float('inf') else 1000
                return new_value

            for self.metric in self.metric_list:
                for self.video in self.videos_list:
                    data =  load_json(json_metrics(self.metric), object_hook=dict)
                    for self.tiling in self.tiling_list:
                        tiling_data = data[self.vid_proj][self.tiling]
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')
                        for self.quality in self.quality_list:
                            qlt_data = tiling_data[self.quality]
                            for self.tile in self.tile_list:
                                tile_data = qlt_data[self.tile]
                                for self.chunk in self.chunk_list:
                                    chunk_data = tile_data[self.chunk]
                                    bucket().append(process(chunk_data))

            if remove_outliers: self.remove_outliers(data_bucket)

            print(f'  Saving  {self.metric}... ', end='')
            save_json(data_bucket, self.data_bucket_file)
            print(f'  Finished.')

        def make_fit(self, overwrite=False):
            print(f'\n\n====== Making Fit - Bins = {self.bins} ======')
            data_bucket = None
            distributions = self.config['distributions']

            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        print(f'  Fitting - {self.metric} {self.proj} {self.tiling}... ', end='')

                        if not overwrite and self.fitter_pickle_file.exists():
                            # Check fitter pickle
                            print(f'Pickle found! Skipping.')
                            continue

                        try:
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                            samples = data_bucket[self.metric][self.proj][self.tiling]

                        # Make a fit
                        fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                        fitter.fit()

                        # Save
                        print(f'  Saving... ', end='')
                        save_pickle(fitter, self.fitter_pickle_file)
                        print(f'  Finished.')

        def calc_stats(self, overwrite=False):
            print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
            data_bucket = None

            if overwrite or not self.stats_file.exists():
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        for self.metric in self.metric_list:
                            # Get samples and Fitter
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                            fitter = load_pickle(self.fitter_pickle_file)

                            # Calculate percentiles
                            percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                            # Calculate errors
                            df_errors: pd.DataFrame = fitter.df_errors
                            sse: pd.Series = df_errors['sumsquare_error']
                            bins = len(fitter.x)
                            rmse = np.sqrt(sse / bins)
                            nrmse = rmse / (sse.max() - sse.min())

                            # Append info and stats on Dataframe
                            self.stats[f'proj'].append(self.proj)
                            self.stats[f'tiling'].append(self.tiling)
                            self.stats[f'metric'].append(self.metric)
                            self.stats[f'bins'].append(self.bins)

                            self.stats[f'average'].append(np.average(samples))
                            self.stats[f'std'].append(float(np.std(samples)))

                            self.stats[f'min'].append(percentile[0])
                            self.stats[f'quartile1'].append(percentile[1])
                            self.stats[f'median'].append(percentile[2])
                            self.stats[f'quartile3'].append(percentile[3])
                            self.stats[f'max'].append(percentile[4])

                            # Append distributions on Dataframe
                            for dist in sse.keys():
                                params = fitter.fitted_param[dist]
                                dist_info = DectimeGraphs.find_dist(dist, params)

                                self.stats[f'rmse_{dist}'].append(rmse[dist])
                                self.stats[f'nrmse_{dist}'].append(nrmse[dist])
                                self.stats[f'sse_{dist}'].append(sse[dist])
                                self.stats[f'param_{dist}'].append(dist_info['parameters'])
                                self.stats[f'loc_{dist}'].append(dist_info['loc'])
                                self.stats[f'scale_{dist}'].append(dist_info['scale'])

                pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
            else:
                print(f'  stats_file found! Skipping.')

            if overwrite or not self.correlations_file.exists():
                if not data_bucket:
                    data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                corretations_bucket = defaultdict(list)

                for metric1, metric2 in combinations(self.metric_list, r=2):
                    for self.proj in self.proj_list:
                        for self.tiling in self.tiling_list:
                            samples1 = data_bucket[metric1][self.proj][self.tiling]
                            samples2 = data_bucket[metric2][self.proj][self.tiling]
                            corrcoef = np.corrcoef((samples1, samples2))[1][0]

                            corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                            corretations_bucket[f'proj'].append(self.proj)
                            corretations_bucket[f'tiling'].append(self.tiling)
                            corretations_bucket[f'corr'].append(corrcoef)

                pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)
            else:
                print(f'  correlations_file found! Skipping.')

        def make_hist(self, overwrite=False):
            print(f'\n====== Make Histogram - Bins = {self.bins} ======')
            folder = self.workfolder / 'pdf_cdf'
            folder.mkdir(parents=True, exist_ok=True)
            subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    im_file = folder / f'pdf_{self.proj}_{self.metric}.png'

                    # Check image file by metric
                    if im_file.exists() and not overwrite:
                        warning(f'Histogram exist. Skipping')
                        continue

                    fig_pdf: figure.Figure = plt.figure(figsize=(12.0, 2))  # pdf
                    fig_cdf: figure.Figure = plt.Figure(figsize=(12.0, 2))  # cdf

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Load fitter
                        fitter = load_pickle(self.fitter_pickle_file)
                        dists = fitter.df_errors['sumsquare_error'].sort_values()[0:self.n_dist].index

                        # <editor-fold desc="Make PDF">
                        # Create a subplot
                        ax: axes.Axes = fig_cdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        ax.bar(fitter.x, fitter.y, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # Make plot for n_dist distributions
                        for dist_name in dists:
                            fitted_pdf = fitter.fitted_pdf[dist_name]
                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'

                            ax.plot(fitter.x, fitted_pdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Density' if index in [1, 6] else None
                        legkwrd = {'loc': 'upper right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                        # <editor-fold desc="Make CDF">
                        ax: axes.Axes = fig_pdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        bins_height = np.cumsum([y * (fitter.x[1] - fitter.x[0]) for y in fitter.y])
                        ax.bar(fitter.x, bins_height, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # make plot for n_dist distributions
                        for dist_name in dists:
                            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                            param = fitter.fitted_param[dist_name]
                            fitted_cdf = dist.cdf(fitter.x, *param)

                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'
                            ax.plot(fitter.x, fitted_cdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Cumulative' if index in [1, 6] else None
                        legkwrd = {'loc': 'lower right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                    print(f'  Saving the PDF')
                    fig_pdf.savefig(im_file)

                    print(f'  Saving the CDF ')
                    im_file = folder / f'cdf_{self.proj}_{self.metric}.png'
                    fig_cdf.savefig(im_file)

        def make_boxplot(self, overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')
            folder = self.workfolder / 'boxplot'
            folder.mkdir(parents=True, exist_ok=True)

            subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            data_bucket = None
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]

            # make an image for each metric and projection
            for mid, self.metric in enumerate(self.metric_list):
                for self.proj in self.proj_list:
                    img_file = folder / f'boxplot_pattern_{mid}{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if img_file.exists() and not overwrite:
                        warning(f'BoxPlot exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        suptitle = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        suptitle = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = None
                        suptitle = self.metric
                    # </editor-fold>

                    fig_boxplot = plt.Figure(figsize=(6., 2.))
                    fig_boxplot.suptitle(f'{suptitle}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        try:
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file, object_hook=AutoDict)
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            tiling_data = [data for data in tiling_data if data < 1000]

                        ax: axes.Axes = fig_boxplot.add_subplot(nrows, ncols, index)
                        boxplot_sep = ax.boxplot((tiling_data,), widths=0.8,
                                                 whis=(0, 100),
                                                 showfliers=False,
                                                 boxprops=dict(facecolor='tab:blue'),
                                                 flierprops=dict(color='r'),
                                                 medianprops=dict(color='k'),
                                                 patch_artist=True)
                        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                        ax.set_xticks([0])
                        ax.set_xticklabels([self.tiling_list[index - 1]])
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                    print(f'  Saving the figure')
                    fig_boxplot.savefig(img_file)

        def make_violinplot(self, overwrite=False):
            print(f'\n====== Make Violin - Bins = {self.bins} ======')
            folder = self.workfolder / 'violinplot'
            folder.mkdir(parents=True, exist_ok=True)

            subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]

            # make an image for each metric and projection
            for mid, self.metric in enumerate(self.metric_list):
                for self.proj in self.proj_list:
                    img_file = folder / f'violinplot_pattern_{mid}{self.metric}_{self.proj}.png'

                    if img_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        title = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        title = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        title = self.metric
                    # </editor-fold>

                    fig = figure.Figure(figsize=(6.8, 3.84))
                    fig.suptitle(f'{title}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        try:
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]
                        except NameError:
                            data_bucket = load_json(self.data_bucket_file)
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            tiling_data = [data for data in tiling_data if data < 1000]

                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax.violinplot([tiling_data], positions=[1],
                                      showmedians=True, widths=0.9)

                        ax.set_xticks([1])
                        ax.set_xticklabels([self.tiling_list[index-1]])
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    class ByPatternByQuality(Worker, DectimeGraphsPaths):
        @property
        def fitter_pickle_file(self) -> Path:
            fitter_file = self.workfolder_data / f'fitter_{self.metric}_{self.proj}_{self.tiling}_{self.quality}_{self.bins}bins.pickle'
            return fitter_file

        def video_context(self): ...

        def __init__(self, config):
            self.config = config
            self.print_resume()
            self.n_dist = 6
            self.bins = 30
            self.stats = defaultdict(list)
            self.corretations_bucket = defaultdict(list)

            self.get_data_bucket(overwrite=False)
            self.make_fit(overwrite=False)
            self.calc_stats(overwrite=False)

            DectimeGraphs.rc_config()
            # self.make_hist(overwrite=True)
            # self.make_bar_tiling_quality(overwrite=True)
            # self.make_bar_quality_tiling(overwrite=True)
            self.make_boxplot(overwrite=True)
            # self.make_violinplot(overwrite=True)

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            # Check file
            if not overwrite and self.data_bucket_file.exists():
                print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                return

            # Get value for each tiling from videos json. tiling = [video, tile, chunk]
            data_bucket = AutoDict()
            json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                           'time': self.dectime_result_json,
                                           'time_std': self.dectime_result_json,
                                           'PSNR': self.quality_result_json,
                                           'WS-PSNR': self.quality_result_json,
                                           'S-PSNR': self.quality_result_json}[metric]

            def bucket():
                # [metric][vid_proj][tiling][quality] = [video, tile, chunk]
                # 1x1 - 1669 chunks/quality
                # 3x2 - 10014 chunks/quality
                # 6x4 - 40056 chunks/quality
                # 9x6 - 90126 chunks/quality
                # 12x8 - 160224 chunks/quality
                # total - 302089 chunks/quality

                data = data_bucket[self.metric][self.vid_proj][self.tiling][self.quality]
                if not isinstance(data, list):
                    data = data_bucket[self.metric][self.vid_proj][self.tiling][self.quality] = []
                return data

            def process(value, remove_inf=True) -> float:
                # Process value according the metric
                if self.metric == 'time':
                    new_value = float(np.average(value['times']))
                elif self.metric == 'time_std':
                    new_value = float(np.std(value['times']))
                elif self.metric == 'rate':
                    new_value = float(value['rate'])
                else:
                    # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                    new_value = float(value[self.metric])
                    if remove_inf:
                        new_value = 1000 if new_value == float('inf') else new_value
                return new_value

            for self.metric in self.metric_list:
                for self.video in self.videos_list:
                    data =  load_json(json_metrics(self.metric), object_hook=dict)

                    for self.tiling in self.tiling_list:
                        tiling_data = data[self.vid_proj][self.tiling]
                        for self.quality in self.quality_list:
                            qlt_data = tiling_data[self.quality]
                            print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} {self.quality} ... ', end='')
                            for self.tile in self.tile_list:
                                tile_data = qlt_data[str(self.tile)]
                                for self.chunk in self.chunk_list:
                                    chunk_data = tile_data[str(self.chunk)]
                                    bucket().append(process(chunk_data))

            if remove_outliers: self.remove_outliers(data_bucket)
            print(f'\n  Saving... ', end='')
            save_json(data_bucket, self.data_bucket_file)
            print(f'  Finished.')

        def make_fit(self, overwrite=False):
            print(f'\n\n====== Making Fit - Bins = {self.bins} ======')
            data_bucket = None
            distributions = self.config['distributions']

            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            print(f'  Fitting - {self.metric} {self.proj} {self.tiling} {self.quality}... ', end='')

                            if not overwrite and self.fitter_pickle_file.exists():
                                # Check fitter pickle
                                print(f'Pickle found! Skipping.')
                                continue

                            try:
                                samples = data_bucket[self.metric][self.proj][self.tiling][self.quality]
                            except TypeError:
                                data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                                samples = data_bucket[self.metric][self.proj][self.tiling][self.quality]

                            # Make a fit
                            fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                            fitter.fit()

                            # Save
                            print(f'  Saving... ', end='')
                            save_pickle(fitter, self.fitter_pickle_file)
                            print(f'  Finished.')

        def calc_stats(self, overwrite=False):
            print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
            data_bucket = None

            if overwrite or not self.stats_file.exists():
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            for self.metric in self.metric_list:
                                # Get samples and Fitter
                                samples = data_bucket[self.metric][self.proj][self.tiling][self.quality]
                                fitter = load_pickle(self.fitter_pickle_file)

                                # Calculate percentiles
                                percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                                # Calculate errors
                                df_errors: pd.DataFrame = fitter.df_errors
                                sse: pd.Series = df_errors['sumsquare_error']
                                bins = len(fitter.x)
                                rmse = np.sqrt(sse / bins)
                                nrmse = rmse / (sse.max() - sse.min())

                                # Append info and stats on Dataframe
                                self.stats[f'proj'].append(self.proj)
                                self.stats[f'tiling'].append(self.tiling)
                                self.stats[f'quality'].append(self.quality)
                                self.stats[f'metric'].append(self.metric)
                                self.stats[f'bins'].append(self.bins)

                                self.stats[f'average'].append(np.average(samples))
                                self.stats[f'std'].append(float(np.std(samples)))

                                self.stats[f'min'].append(percentile[0])
                                self.stats[f'quartile1'].append(percentile[1])
                                self.stats[f'median'].append(percentile[2])
                                self.stats[f'quartile3'].append(percentile[3])
                                self.stats[f'max'].append(percentile[4])

                                # Append distributions on Dataframe
                                for dist in sse.keys():
                                    params = fitter.fitted_param[dist]
                                    dist_info = DectimeGraphs.find_dist(dist, params)

                                    self.stats[f'rmse_{dist}'].append(rmse[dist])
                                    self.stats[f'nrmse_{dist}'].append(nrmse[dist])
                                    self.stats[f'sse_{dist}'].append(sse[dist])
                                    self.stats[f'param_{dist}'].append(dist_info['parameters'])
                                    self.stats[f'loc_{dist}'].append(dist_info['loc'])
                                    self.stats[f'scale_{dist}'].append(dist_info['scale'])

                pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
            else:
                print(f'  stats_file found! Skipping.')

            if overwrite or not self.correlations_file.exists():
                if not data_bucket:
                    data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                corretations_bucket = defaultdict(list)

                for metric1, metric2 in combinations(self.metric_list, r=2):
                    for self.proj in self.proj_list:
                        for self.tiling in self.tiling_list:
                            for self.quality in self.quality_list:
                                samples1 = data_bucket[metric1][self.proj][self.tiling][self.quality]
                                samples2 = data_bucket[metric2][self.proj][self.tiling][self.quality]
                                corrcoef = np.corrcoef((samples1, samples2))[1][0]

                                corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                                corretations_bucket[f'proj'].append(self.proj)
                                corretations_bucket[f'tiling'].append(self.tiling)
                                corretations_bucket[f'quality'].append(self.quality)
                                corretations_bucket[f'corr'].append(corrcoef)

                pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)
            else:
                print(f'  correlations_file found! Skipping.')

        def make_hist(self, overwrite=False):
            """
            1-Um arquivo por tiling. Cada subplot é uma qualidade
            2-Um aruivo por qualidade. Cada subplot é um tiling
            OBS. Cosniderar como erro o SSE e o BIC.

            :param overwrite:
            :return:
            """
            print(f'\n====== Make hist - Bins = {self.bins} ======')
            folder = self.workfolder / 'pdf_cdf_by_tiling'
            # folder = self.workfolder / 'pdf_cdf_by_quality'
            folder.mkdir(parents=True, exist_ok=True)
            subplot_pos = [(1, 6, x) for x in range(1, 7)]  # 1x5
            error_type = 'sse'  # 'sse' or 'bic'

            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    # for self.quality in self.quality_list:
                    for self.tiling in self.tiling_list:
                        # Check image file by metric
                        # im_file = folder / f'pdf_{self.proj}_{self.metric}_{self.quality}.png'
                        im_file = folder / f'pdf_{self.proj}_{self.metric}_{self.tiling}_{error_type}.png'
                        if im_file.exists() and not overwrite:
                            warning(f'Figure PDF exist. Skipping')
                            continue
                        else:
                            # Make figure
                            fig_pdf: figure.Figure = plt.figure(figsize=(12.0, 2))  # pdf

                            # for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                            for self.quality, (nrows, ncols, index) in zip(self.quality_list, subplot_pos):
                                # Load fitter and select samples
                                fitter = load_pickle(self.fitter_pickle_file)
                                # Bayesian Information Criterion (BIC) - http://www.ime.unicamp.br/sinape/sites/default/files/Paulo%20C%C3%A9sar%20Emiliano.pdf
                                error_key = 'sumsquare_error' if error_type == 'sse' else 'bic'
                                dists = fitter.df_errors[error_key].sort_values()[0:self.n_dist].index

                                # <editor-fold desc="Make PDF">
                                # Create a subplot
                                ax: axes.Axes = fig_pdf.add_subplot(nrows, ncols, index)

                                # Make bars of histogram
                                ax.bar(fitter.x, fitter.y, label='empirical', color='#dbdbdb',
                                       width=fitter.x[1] - fitter.x[0])

                                # Make plot for n_dist distributions
                                for dist_name in dists:
                                    fitted_pdf = fitter.fitted_pdf[dist_name]
                                    error = fitter.df_errors[error_key][dist_name]
                                    label = f'{dist_name} - {error_type.upper()} {error: 0.3e}'

                                    ax.plot(fitter.x, fitted_pdf,
                                            color=self.color_list[dist_name], label=label)

                                # <editor-fold desc="Format plot">
                                title = f'{self.tiling} - CRF {self.quality}'
                                ylabel = 'Density' if index in [1, 6] else None
                                legkwrd = {'loc': 'upper right'}

                                if self.metric == 'time':
                                    scilimits = (-3, -3)
                                    xlabel = f'Decoding Time (ms)'
                                elif self.metric == 'time_std':
                                    scilimits = (-3, -3)
                                    xlabel = f'Std Dev - Decoding Time (ms)'
                                elif self.metric == 'rate':
                                    scilimits = (6, 6)
                                    xlabel = f'Bit Rate (Mbps)'
                                else:
                                    scilimits = (0, 0)
                                    xlabel = self.metric

                                ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                                ax.set_title(title)
                                ax.set_xlabel(xlabel)
                                ax.set_ylabel(ylabel)
                                # ax.set_yscale('log')
                                ax.legend(**legkwrd)
                                # </editor-fold>

                                # </editor-fold>

                            print(f'  Saving the PDF')
                            fig_pdf.savefig(im_file)

                        # CDF
                        stem = im_file.stem.replace('cdf', 'pdf')
                        im_file = im_file.with_stem(stem)
                        if im_file.exists() and not overwrite:
                            warning(f'Figure CDF exist. Skipping')
                            continue
                        else:
                            # Make figure
                            fig_cdf: figure.Figure = plt.Figure(figsize=(12.0, 2))  # cdf

                            # for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                            for self.quality, (nrows, ncols, index) in zip(self.quality_list, subplot_pos):
                                # Load fitter and select samples
                                fitter = load_pickle(self.fitter_pickle_file)
                                # Bayesian Information Criterion (BIC) - http://www.ime.unicamp.br/sinape/sites/default/files/Paulo%20C%C3%A9sar%20Emiliano.pdf
                                error_key = 'sumsquare_error' if error_type == 'sse' else 'bic'
                                dists = fitter.df_errors[error_key].sort_values()[0:self.n_dist].index

                                # <editor-fold desc="Make CDF">
                                ax: axes.Axes = fig_cdf.add_subplot(nrows, ncols, index)

                                # Make bars of histogram
                                bins_height = np.cumsum([y * (fitter.x[1] - fitter.x[0]) for y in fitter.y])
                                ax.bar(fitter.x, bins_height, label='empirical', color='#dbdbdb',
                                       width=fitter.x[1] - fitter.x[0])

                                # make plot for n_dist distributions
                                for dist_name in dists:
                                    dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                                    param = fitter.fitted_param[dist_name]
                                    fitted_cdf = dist.cdf(fitter.x, *param)
                                    error = fitter.df_errors[error_key][dist_name]
                                    label = f'{dist_name} - {error_type.upper()} {error: 0.3e}'
                                    ax.plot(fitter.x, fitted_cdf, color=self.color_list[dist_name], label=label)

                                # <editor-fold desc="Format plot">
                                title = f'{self.proj.upper()}-{self.tiling}'
                                ylabel = 'Cumulative' if index in [1, 6] else None
                                legkwrd = {'loc': 'lower right'}

                                if self.metric == 'time':
                                    scilimits = (-3, -3)
                                    xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                                elif self.metric == 'time_std':
                                    scilimits = (-3, -3)
                                    xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                                elif self.metric == 'rate':
                                    scilimits = (6, 6)
                                    xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                                else:
                                    scilimits = (0, 0)
                                    xlabel = self.metric

                                ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                                ax.set_title(title)
                                ax.set_xlabel(xlabel)
                                ax.set_ylabel(ylabel)
                                # ax.set_yscale('log')
                                ax.legend(**legkwrd)
                                # </editor-fold>

                                # </editor-fold>

                            print(f'  Saving the CDF ')
                            fig_cdf.savefig(im_file)
                            plt.close('all')

        def make_bar_tiling_quality(self, overwrite=False):
            """
            fig = metric
            subplot = tiling (5)
            bar = quality (6)
            """
            print(f'\n====== Make bar_tiling_quality ======')
            folder = self.workfolder / 'bar_tiling_quality'
            folder.mkdir(parents=True, exist_ok=True)

            stats = None
            x = xticks = list(range(len(self.quality_list)))
            colors = {'time': 'tab:blue', 'rate': 'tab:red'}
            legend_handles = [mpatches.Patch(color=colors['time'], label='Time'),
                              mlines.Line2D([], [], color=colors['rate'], label='Bitrate')]

            for mid, self.metric in enumerate(self.metric_list):
                for self.proj in self.proj_list:
                    im_file = folder / f'bar_{mid}{self.metric}_tiling_quality.png'
                    if im_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        return

                    # Make figure
                    fig: figure.Figure = plt.Figure(figsize=(12., 2.))
                    subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Bar plot of dectime
                        try:
                            data = stats[(stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == self.metric)]
                        except TypeError:
                            stats = pd.read_csv(self.stats_file)
                            data = stats[(stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == self.metric)]

                        height = data[f'average']
                        yerr = data[f'std']

                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax.bar(x, height, width=0.8, yerr=yerr, color=colors['time'])

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()} - {self.tiling}'
                        xlabel = 'CRF'

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            ylabel = f'Average Decoding Time (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            ylabel = f'Std Dev - Decoding Time (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            ylabel = f'Bit Rate (Mbps)'
                        else:
                            scilimits = (0, 0)
                            ylabel = self.metric

                        ax.set_title(title)
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(self.quality_list)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        ax.set_xlabel(xlabel)
                        if index in [1, 6]: ax.set_ylabel(ylabel)
                        ax.legend(handles=legend_handles, loc='upper right')
                        # </editor-fold>

                        ### Line plot of bit rate
                        if self.metric in ['time_std', 'rate']: continue
                        data = stats[(stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'rate')]
                        rate_avg = data['average']
                        rate_stdp = rate_avg - data['std']
                        rate_stdm = rate_avg + data['std']

                        ax: axes.Axes = ax.twinx()
                        ax.plot(x, rate_avg, color=colors['rate'], linewidth=1)
                        ax.plot(x, rate_stdp, color='gray', linewidth=1)
                        ax.plot(x, rate_stdm, color='gray', linewidth=1)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(6, 6))
                        if index in [5]: ax.set_ylabel(f'Bit Rate (Mbps)')

                    print(f'Salvando a figura')
                    fig.savefig(im_file)

        def make_boxplot(self, overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')
            folder = self.workfolder / 'boxplot'
            folder.mkdir(parents=True, exist_ok=True)

            subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            data_bucket = None
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]
            x = xticks = list(range(len(self.quality_list)))

            # make an image for each metric and projection
            for mid, self.metric in enumerate(self.metric_list):
                for self.proj in self.proj_list:
                    img_file = folder / f'boxplot_pattern_{mid}{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if img_file.exists() and not overwrite:
                        warning(f'BoxPlot exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    xlabel = 'CRF'
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        ylabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        ylabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        ylabel = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        ylabel = self.metric
                    # </editor-fold>

                    fig = plt.Figure(figsize=(12., 2.))

                    # for self.quality in self.quality_list:
                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        try:
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file, object_hook=AutoDict)
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                        data = []
                        for self.quality in self.quality_list:
                            quality_data = tiling_data[self.quality]
                            if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                quality_data = [data for data in quality_data if data < 1000]
                            data.append(quality_data)

                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        boxplot_sep = ax.boxplot(data, positions=x, widths=0.8,
                                                 # whis=(0, 100),
                                                 showfliers=False,
                                                 boxprops=dict(facecolor='tab:blue'),
                                                 flierprops=dict(color='r'),
                                                 medianprops=dict(color='k'),
                                                 patch_artist=True)
                        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                        title = f'{self.proj.upper()} - {self.tiling}'
                        ax.set_title(title)
                        ax.set_xticks(x)
                        ax.set_xlabel(xlabel)
                        ax.set_xticklabels(self.quality_list)
                        if index in [1]: ax.set_ylabel(ylabel)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        def make_violinplot(self, overwrite=False):
            print(f'\n====== Make Violin - Bins = {self.bins} ======')
            folder = self.workfolder / 'violinplot'
            folder.mkdir(parents=True, exist_ok=True)
            data_bucket = None
            subplot_pos = [(1, 5, x) for x in range(1, 6)]  # 1x5
            x = xticks = list(range(len(self.quality_list)))

            # make an image for each metric and projection
            for mid, self.metric in enumerate(self.metric_list):
                for self.proj in self.proj_list:
                    img_file = folder / f'violinplot_pattern_{mid}{self.metric}_{self.proj}.png'

                    if img_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    xlabel = 'CRF'
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        ylabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        ylabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        ylabel = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        ylabel = self.metric
                    # </editor-fold>

                    fig = plt.Figure(figsize=(12., 2.))

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        try:
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file)
                            tiling_data = data_bucket[self.metric][self.proj][self.tiling]

                        data = []
                        for self.quality in self.quality_list:
                            quality_data = tiling_data[self.quality]
                            if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                quality_data = [data for data in quality_data if data < 1000]
                            data.append(quality_data)

                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax.violinplot(data, positions=x, showmedians=True, widths=0.9)

                        title = f'{self.proj.upper()} - {self.tiling}'
                        ax.set_title(title)
                        ax.set_xticks(x)
                        ax.set_xlabel(xlabel)
                        ax.set_xticklabels(self.quality_list)
                        if index in [1]: ax.set_ylabel(ylabel)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    class ByVideoByPatternByQualityByTile(Worker, DectimeGraphsPaths):
        ''' Série temporal
        cada imagem é um vídeo, cada subplot é um tiling.
        No mesmo subplot, cada linha é um tile.
        Os chunks é o eixo X.
        resultado: subplot = 5x1

        '''
        dists_colors = {'burr12': 'tab:blue',
                        'fatiguelife': 'tab:orange',
                        'gamma': 'tab:green',
                        'invgauss': 'tab:red',
                        'rayleigh': 'tab:purple',
                        'lognorm': 'tab:brown',
                        'genpareto': 'tab:pink',
                        'pareto': 'tab:gray',
                        'halfnorm': 'tab:olive',
                        'expon': 'tab:cyan'}

        @property
        def workfolder(self) -> Path:
            folder = self.project_path / self.graphs_folder / 'ByVideoByPatternByQualityByTile'
            folder.mkdir(parents=True, exist_ok=True)
            return folder

        @property
        def workfolder_data(self) -> Path:
            folder = self.workfolder / 'data'
            folder.mkdir(parents=True, exist_ok=True)
            return folder

        def __init__(self, config):
            self.config = config
            self.print_resume()

            self.bins = 30
            self.stats: dict[str, list] = defaultdict(list)
            self.corretations_bucket = defaultdict(list)
            self.data: Optional[Union[dict, AutoDict]] = None

            self.get_data_bucket()
            # self.make_fit()
            # self.calc_stats()

            DectimeGraphs.rc_config()
            self.make_plot_series(overwrite=True)
            # self.make_boxplot(overwrite=True)
            # self.make_violinplot(overwrite=True)

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            # Check file
            if self.data_bucket_file.exists() and not overwrite:
                print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                return

            # Get value for each tiling from videos json. tiling = [video, tile, chunk]
            data_bucket = AutoDict()
            json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                           'time': self.dectime_result_json,
                                           'time_std': self.dectime_result_json,
                                           'PSNR': self.quality_result_json,
                                           'WS-PSNR': self.quality_result_json,
                                           'S-PSNR': self.quality_result_json}[metric]

            def bucket():
                # [metric][vid_proj][tiling][quality] = [video, tile, chunk]
                # 1x1 - 1669 chunks/quality
                # 3x2 - 10014 chunks/quality
                # 6x4 - 40056 chunks/quality
                # 9x6 - 90126 chunks/quality
                # 12x8 - 160224 chunks/quality
                # total - 302089 chunks/quality

                data = data_bucket[self.metric][self.vid_proj][self.tiling][self.tile]
                if not isinstance(data, list):
                    data = data_bucket[self.metric][self.vid_proj][self.tiling][self.tile] = []
                return data

            def process(value, remove_inf=True) -> float:
                # Process value according the metric
                if self.metric == 'time':
                    new_value = float(np.average(value['times']))
                elif self.metric == 'time_std':
                    new_value = float(np.std(value['times']))
                elif self.metric == 'rate':
                    new_value = float(value['rate'])
                else:
                    # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                    new_value = float(value[self.metric])
                    if remove_inf:
                        new_value = 1000 if new_value == float('inf') else new_value
                return new_value

            for self.metric in self.metric_list:
                for self.video in self.videos_list:
                    data =  load_json(json_metrics(self.metric), object_hook=dict)
                    for self.tiling in self.tiling_list:
                        tiling_data = data[self.vid_proj][self.tiling]
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')

                        for self.tile in self.tile_list:

                            for self.chunk in self.chunk_list:
                                chunk_qualities = []
                                for self.quality in self.quality_list:
                                    value = tiling_data[self.quality][self.tile][self.chunk]

                                    chunk_qualities.append(process(value))

                                value = np.average(chunk_qualities)
                                bucket().append()

                        print('OK')

                if remove_outliers: self.remove_outliers(data_bucket)

                print(f'  Saving  {self.metric}... ', end='')
                save_json(data_bucket, self.data_bucket_file)
                print(f'  Finished.')

        def calc_stats(self, overwrite=False):
            print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
            if overwrite or not self.stats_file.exists():
                for self.metric in self.metric_list:
                    data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                    for self.proj in self.proj_list:
                        for self.video in self.videos_list:
                            for self.tiling in self.tiling_list:
                                for self.tile in self.tile_list:
                                    # Load data and fitter
                                    samples = data_bucket[self.proj][self.video][self.tiling][self.tile]

                                    # Calculate percentiles
                                    percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                                    # Append info and stats on Dataframe
                                    self.stats[f'proj'].append(self.proj)
                                    self.stats[f'tiling'].append(self.tiling)
                                    self.stats[f'metric'].append(self.metric)
                                    self.stats[f'average'].append(np.average(samples))
                                    self.stats[f'std'].append(float(np.std(samples)))
                                    self.stats[f'min'].append(percentile[0])
                                    self.stats[f'quartile1'].append(percentile[1])
                                    self.stats[f'median'].append(percentile[2])
                                    self.stats[f'quartile3'].append(percentile[3])
                                    self.stats[f'max'].append(percentile[4])

                pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
            # Check result file
            else:
                print(f'  stats_file found! Skipping.')

            if self.correlations_file.exists() and not overwrite:
                print(f'  correlations_file found! Skipping.')
            else:
                from itertools import combinations
                corretations_bucket = defaultdict(list)
                for metric1, metric2 in combinations(self.metric_list, r=2):
                    self.metric = metric1
                    data_bucket1 = load_json(self.data_bucket_file, object_hook=dict)
                    self.metric = metric2
                    data_bucket2 = load_json(self.data_bucket_file, object_hook=dict)

                    for self.proj in self.proj_list:
                        for self.video in self.videos_list:
                            for self.tiling in self.tiling_list:
                                for self.tile in self.tile_list:
                                    samples1 = data_bucket1[self.proj][self.tiling]
                                    samples2 = data_bucket2[self.proj][self.tiling]
                                    corrcoef = np.corrcoef((samples1, samples2))[1][0]

                                    corretations_bucket[f'proj'].append(self.proj)
                                    corretations_bucket[f'tiling'].append(self.tiling)
                                    corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                                    corretations_bucket[f'corr'].append(corrcoef)

                pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)

        def make_plot_series(self, overwrite=False):
            print(f'\n====== Make PlotSeries - Bins = {self.bins} ======')
            folder = self.workfolder / 'plot_series'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]
            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                for self.video in self.videos_list:
                    print(f'\r  {self.metric} - {self.vid_proj}  {self.name} ... ', end='')
                    img_file = folder / f'plot_{self.vid_proj}_{self.name}_{self.metric}.png'
                    # Check image file by metric
                    if img_file.exists() and not overwrite:
                        warning(f'BoxPlot exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        suptitle = f'Average Decoding {self.metric.capitalize()} (ms) - {self.name} ({self.vid_proj})'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms) - {self.name} ({self.vid_proj})'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        suptitle = f'Bit {self.metric.capitalize()} (Mbps) - {self.name} ({self.vid_proj})'
                    else:
                        scilimits = None
                        suptitle = f'{self.metric} - {self.name} ({self.vid_proj})'
                    # </editor-fold>

                    fig_boxplot = plt.Figure(figsize=(12., 2.))
                    fig_boxplot.suptitle(f'{suptitle}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        ax: axes.Axes = fig_boxplot.add_subplot(nrows, ncols, index)

                        for self.tile in self.tile_list:
                            # make a line

                            # Get data
                            tiling_data = data_bucket[self.vid_proj][self.video][self.tiling][self.tile]

                            if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                                tiling_data = [data for data in tiling_data if data < 1000]

                            ax.plot(tiling_data, label=f'Tile {self.tile}')

                            # if index in [columns]:
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig_boxplot.savefig(img_file)

        def make_violinplot(self, overwrite=False):
            print(f'\n====== Make Violin - Bins = {self.bins} ======')
            folder = self.workfolder / 'violinplot'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file)
                for self.proj in self.proj_list:
                    img_file = folder / f'violinplot_pattern_{self.metric}_{self.proj}.png'

                    if img_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        title = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        title = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        title = self.metric
                    # </editor-fold>

                    fig = figure.Figure(figsize=(6.8, 3.84))
                    fig.suptitle(f'{title}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        tiling_data = data_bucket[self.proj][self.tiling]

                        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            tiling_data = [data for data in tiling_data if data < 1000]

                        ax_sep: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax_sep.violinplot([tiling_data], positions=[1],
                                          showmedians=True, widths=0.9)

                        ax_sep.set_xticks([1])
                        # ax_sep.set_ylim(bottom=0)
                        ax_sep.set_xticklabels([self.tiling_list[index-1]])
                        ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # if index in [columns]:
                        #     ax_sep.legend(handles=legend_handles, loc='upper left',
                        #                   bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    class ByVideoByPatternByTileByQualityByChunk(Worker, DectimeGraphsPaths):
        ''' Série temporal 1
        cada imagem é uma métrica de um vídeo com um tiling,
        cada subplot é um tile.
        No mesmo subplot, cada linha é uma qualidade.
        O subplot tem o formato do tiling
        Os chunks é o eixo X.
        resultado: subplot = 5x1

        -------- 2
        cada imagem é um vídeo e um tiling,
        cada subplot é uma qualidade.
        No mesmo subplot, cada linha é um tile.
        o subplot tem o formato 5x1
        Os chunks é o eixo X.
        resultado: subplot = 5x1


        '''

        dists_colors = {'burr12': 'tab:blue',
                        'fatiguelife': 'tab:orange',
                        'gamma': 'tab:green',
                        'invgauss': 'tab:red',
                        'rayleigh': 'tab:purple',
                        'lognorm': 'tab:brown',
                        'genpareto': 'tab:pink',
                        'pareto': 'tab:gray',
                        'halfnorm': 'tab:olive',
                        'expon': 'tab:cyan'}

        @property
        def workfolder(self) -> Path:
            folder = self.project_path / self.graphs_folder / f'{self.__class__.__name__}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder

        @property
        def workfolder_data(self) -> Path:
            folder = self.workfolder / 'data'
            folder.mkdir(parents=True, exist_ok=True)
            return folder

        def __init__(self, config):
            self.config = config
            self.print_resume()

            self.bins = 30
            self.stats: dict[str, list] = defaultdict(list)
            self.corretations_bucket = defaultdict(list)
            self.data: Optional[Union[dict, AutoDict]] = None

            self.get_data_bucket(overwrite=False)
            # self.make_fit()
            # self.calc_stats()

            DectimeGraphs.rc_config()
            self.make_plot_series1(overwrite=True)
            # self.make_plot_series2(overwrite=True)
            # self.make_boxplot(overwrite=True)
            # self.make_violinplot(overwrite=True)

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            for self.metric in self.metric_list:
                # Check file
                if self.data_bucket_file.exists() and not overwrite:
                    print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                    continue

                # Get value for each tiling from videos json. tiling = [video, quality, tile, chunk]
                data_result = AutoDict()
                for self.video in self.videos_list:
                    json_metrics = {'rate': self.bitrate_result_json,
                                    'time': self.dectime_result_json,
                                    'time_std': self.dectime_result_json,
                                    'PSNR': self.quality_result_json,
                                    'WS-PSNR': self.quality_result_json,
                                    'S-PSNR': self.quality_result_json}
                    video_data =  load_json(json_metrics[self.metric], object_hook=dict)

                    for self.tiling in self.tiling_list:
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} {self.quality} ... ', end='')

                        for self.quality in self.quality_list:
                            for self.tile in self.tile_list:
                                for self.chunk in self.chunk_list:
                                    samples_result = data_result[self.vid_proj][self.name][self.tiling][self.quality][self.tile]
                                    value = video_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']

                                    # <editor-fold desc="Process value according the metric">
                                    if self.metric == 'time':
                                        value = float(np.average(value['times']))
                                    elif self.metric == 'time_std':
                                        value = float(np.std(value['times']))
                                    elif self.metric == 'rate':
                                        value = float(value['rate'])
                                    elif self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                        metric_value = value[self.metric]
                                        value = metric_value if float(metric_value) != float('inf') else 1000
                                    # </editor-fold>

                                    try:
                                        samples_result.append(value)
                                    except AttributeError:
                                        data_result[self.vid_proj][self.name][self.tiling][self.quality][self.tile] = [value]

                        print('OK')

                if remove_outliers: self.remove_outliers(data_result)

                print(f'  Saving  {self.metric}... ', end='')
                save_json(data_result, self.data_bucket_file)
                print(f'  Finished.')

        # <editor-fold desc="calc_stats">
        # def calc_stats(self, overwrite=False):
        #     print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
        #     # Check result file
        #     if self.stats_file.exists() and not overwrite:
        #         print(f'  stats_file found! Skipping.')
        #     else:
        #         for self.metric in self.metric_list:
        #             data_bucket = load_json(self.data_bucket_file, object_hook=dict)
        #             for self.proj in self.proj_list:
        #                 for self.video in self.videos_list:
        #                     for self.tiling in self.tiling_list:
        #                         for self.tile in self.tile_list:
        #                             # Load data and fitter
        #                             samples = data_bucket[self.proj][self.video][self.tiling][self.tile]
        #                             fitter = load_pickle(self.fitter_pickle_file)
        #
        #                             # Calculate percentiles
        #                             percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T
        #                             df_errors: pd.DataFrame = fitter.df_errors
        #
        #                             # Append info and stats on Dataframe
        #                             self.stats[f'proj'].append(self.proj)
        #                             self.stats[f'tiling'].append(self.tiling)
        #                             self.stats[f'metric'].append(self.metric)
        #                             self.stats[f'average'].append(np.average(samples))
        #                             self.stats[f'std'].append(float(np.std(samples)))
        #                             self.stats[f'min'].append(percentile[0])
        #                             self.stats[f'quartile1'].append(percentile[1])
        #                             self.stats[f'median'].append(percentile[2])
        #                             self.stats[f'quartile3'].append(percentile[3])
        #                             self.stats[f'max'].append(percentile[4])
        #
        #         pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
        #
        #     if self.correlations_file.exists() and not overwrite:
        #         print(f'  correlations_file found! Skipping.')
        #     else:
        #         from itertools import combinations
        #         corretations_bucket = defaultdict(list)
        #         for metric1, metric2 in combinations(self.metric_list, r=2):
        #             self.metric = metric1
        #             data_bucket1 = load_json(self.data_bucket_file, object_hook=dict)
        #             self.metric = metric2
        #             data_bucket2 = load_json(self.data_bucket_file, object_hook=dict)
        #
        #             for self.proj in self.proj_list:
        #                 for self.video in self.videos_list:
        #                     for self.tiling in self.tiling_list:
        #                         for self.tile in self.tile_list:
        #                             samples1 = data_bucket1[self.proj][self.tiling]
        #                             samples2 = data_bucket2[self.proj][self.tiling]
        #                             corrcoef = np.corrcoef((samples1, samples2))[1][0]
        #
        #                             corretations_bucket[f'proj'].append(self.proj)
        #                             corretations_bucket[f'tiling'].append(self.tiling)
        #                             corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
        #                             corretations_bucket[f'corr'].append(corrcoef)
        #
        #         pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)
        # </editor-fold>


        def make_plot_series1(self, overwrite=False):
            print(f'\n====== Make PlotSeries - Bins = {self.bins} ======')
            folder = self.workfolder / 'plot_series1'
            folder.mkdir(parents=True, exist_ok=True)

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                metricid = self.metric_list.index(self.metric)
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                for self.video in self.videos_list:
                    for self.tiling in self.tiling_list:
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')
                        img_file = folder / f'plot_{self.vid_proj}_{self.name}_{self.tiling}_{metricid}-{self.metric}.png'

                        # Check image file by metric
                        if img_file.exists() and not overwrite:
                            warning(f'BoxPlot exist. Skipping')
                            continue

                        # <editor-fold desc="Format plot">
                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            suptitle = f'Average Decoding {self.metric.capitalize()} (ms) - {self.name} {self.tiling} ({self.vid_proj})'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms) - {self.name} {self.tiling} ({self.vid_proj})'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            suptitle = f'Bit {self.metric.capitalize()} (Mbps) - {self.name} {self.tiling} ({self.vid_proj})'
                        else:
                            scilimits = None
                            suptitle = f'{self.metric} - {self.name} {self.tiling} ({self.vid_proj})'
                        # </editor-fold>

                        major = 0
                        minor = float('inf')

                        nrows, ncols = splitx(self.tiling)
                        fig = plt.Figure(figsize=(2.*nrows, 2.*ncols), dpi=200)
                        fig.suptitle(f'{suptitle}')

                        for index, self.tile in enumerate(self.tile_list, 1):
                            ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                            for self.quality in self.quality_list:
                                # Get data
                                tiling_data = data_bucket[self.vid_proj][self.name][self.tiling][self.quality][self.tile]

                                # <editor-fold desc="find max-min">
                                a=max(tiling_data)
                                if a > major:
                                    major = a
                                a=min(tiling_data)
                                if a < minor:
                                    minor = a
                                # </editor-fold>

                                # if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                                #     tiling_data = [data for data in tiling_data if data < 1000]

                                ax.plot(tiling_data, label=f'CRF {self.quality}')

                            # if index in [columns]:
                            ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                            ax.legend(loc='upper right', fontsize='small')

                        for ax in fig.axes:
                            ax.set_ylim(ymin=minor, ymax=major)

                        print(f'  Saving the figure')
                        fig.tight_layout()
                        fig.savefig(img_file)
        #
        # def make_plot_series2(self, overwrite=False):
        #     print(f'\n====== Make PlotSeries - Bins = {self.bins} ======')
        #     folder = self.workfolder / 'plot_series'
        #     folder.mkdir(parents=True, exist_ok=True)
        #
        #     row, columns = 1, 5
        #     subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
        #     colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
        #     legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
        #                       # mpatches.Patch(color=colors['cmp'], label='CMP'),
        #                       ]
        #     # make an image for each metric and projection
        #     for self.metric in self.metric_list:
        #         data_bucket = load_json(self.data_bucket_file, object_hook=dict)
        #
        #         for self.video in self.videos_list:
        #             print(f'\r  {self.metric} - {self.vid_proj}  {self.name} ... ', end='')
        #             img_file = folder / f'plot_{self.vid_proj}_{self.name}_{self.metric}.png'
        #             # Check image file by metric
        #             if img_file.exists() and not overwrite:
        #                 warning(f'BoxPlot exist. Skipping')
        #                 continue
        #
        #             # <editor-fold desc="Format plot">
        #             if self.metric == 'time':
        #                 scilimits = (-3, -3)
        #                 suptitle = f'Average Decoding {self.metric.capitalize()} (ms) - {self.name} ({self.vid_proj})'
        #             elif self.metric == 'time_std':
        #                 scilimits = (-3, -3)
        #                 suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms) - {self.name} ({self.vid_proj})'
        #             elif self.metric == 'rate':
        #                 scilimits = (6, 6)
        #                 suptitle = f'Bit {self.metric.capitalize()} (Mbps) - {self.name} ({self.vid_proj})'
        #             else:
        #                 scilimits = None
        #                 suptitle = f'{self.metric} - {self.name} ({self.vid_proj})'
        #             # </editor-fold>
        #
        #             fig_boxplot = plt.Figure(figsize=(12., 2.))
        #             fig_boxplot.suptitle(f'{suptitle}')
        #
        #             for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
        #                 ax: axes.Axes = fig_boxplot.add_subplot(nrows, ncols, index)
        #
        #                 for self.tile in self.tile_list:
        #                     # make a line
        #
        #                     # Get data
        #                     tiling_data = data_bucket[self.vid_proj][self.video][self.tiling][self.tile]
        #
        #                     if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
        #                         tiling_data = [data for data in tiling_data if data < 1000]
        #
        #                     ax.plot(tiling_data, label=f'Tile {self.tile}')
        #
        #                     # if index in [columns]:
        #                 ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
        #                 # ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')
        #
        #             print(f'  Saving the figure')
        #             fig_boxplot.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    @staticmethod
    def find_dist(dist_name, params):
        if dist_name == 'burr12':
            return dict(name='Burr Type XII',
                        parameters=f'c={params[0]}, d={params[1]}',
                        loc=params[2],
                        scale=params[3])
        elif dist_name == 'fatiguelife':
            return dict(name='Birnbaum-Saunders',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'gamma':
            return dict(name='Gamma',
                        parameters=f'a={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'invgauss':
            return dict(name='Inverse Gaussian',
                        parameters=f'mu={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'rayleigh':
            return dict(name='Rayleigh',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'lognorm':
            return dict(name='Log Normal',
                        parameters=f's={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'genpareto':
            return dict(name='Generalized Pareto',
                        parameters=f'c={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'pareto':
            return dict(name='Pareto Distribution',
                        parameters=f'b={params[0]}',
                        loc=params[1],
                        scale=params[2])
        elif dist_name == 'halfnorm':
            return dict(name='Half-Normal',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        elif dist_name == 'expon':
            return dict(name='Exponential',
                        parameters=f' ',
                        loc=params[0],
                        scale=params[1])
        else:
            raise ValueError(f'Distribution unknown: {dist_name}')

    @staticmethod
    def rc_config():
        import matplotlib as mpl
        rc_param = {"figure": {'figsize': (7.0, 1.2), 'dpi': 600, 'autolayout': True},
                    "axes": {'linewidth': 0.5, 'titlesize': 8, 'labelsize': 6,
                             'prop_cycle': cycler(color=[plt.get_cmap('tab20')(i) for i in range(20)])},
                    "xtick": {'labelsize': 6},
                    "ytick": {'labelsize': 6},
                    "legend": {'fontsize': 6},
                    "font": {'size': 6},
                    "patch": {'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': '#3297c9'},
                    "lines": {'linewidth': 0.5, 'markersize': 2},
                    "errorbar": {'capsize': 4},
                    "boxplot": {'flierprops.marker': '+', 'flierprops.markersize': 1, 'flierprops.linewidth': 0.5,
                                'boxprops.linewidth': 0.0,
                                'capprops.linewidth': 1,
                                'medianprops.linewidth': 0.5,
                                'whiskerprops.linewidth': 0.5,
                                }
                    }

        for group in rc_param:
            mpl.rc(group, **rc_param[group])

    # <editor-fold desc="by_pattern">




    # def by_pattern_full_frame(self, overwrite):
    #     class ProjectPaths:
    #         ctx = self
    #
    #         @property
    #         def quality_file(self) -> Path:
    #             quality_file = self.ctx.video_context.project_path / self.ctx.video_context.quality_folder / f'quality_{self.ctx.video}.json'
    #             return quality_file
    #
    #         @property
    #         def dectime_file(self) -> Path:
    #             dectime_file = self.ctx.video_context.project_path / self.ctx.video_context.dectime_folder / f'dectime_{self.ctx.video}.json'
    #             return dectime_file
    #
    #         @property
    #         def data_file(self) -> Path:
    #             data_file = self.ctx.workfolder_data / f'data.json'
    #             return data_file
    #
    #         @property
    #         def fitter_pickle_file(self) -> Path:
    #             fitter_file = self.ctx.workfolder_data / f'fitter_{self.ctx.proj}_{self.ctx.tiling}_{self.ctx.metric}_{self.ctx.bins}bins.pickle'
    #             return fitter_file
    #
    #         @property
    #         def stats_file(self) -> Path:
    #             stats_file = self.ctx.workfolder / f'stats_{self.ctx.bins}bins.csv'
    #             return stats_file
    #
    #         @property
    #         def corretations_file(self) -> Path:
    #             corretations_file = self.ctx.workfolder / f'correlations.csv'
    #             return corretations_file
    #
    #         @property
    #         def boxplot_pattern_full_frame_file(self) -> Path:
    #             img_file = self.ctx.workfolder / f'boxplot_pattern_full_frame_{self.ctx.tiling}_{self.ctx.metric}.png'
    #             return img_file
    #
    #         @property
    #         def hist_pattern_full_frame_file(self) -> Path:
    #             img_file = self.ctx.workfolder / f'hist_pattern_full_frame_{self.ctx.metric}_{self.ctx.bins}bins.png'
    #             return img_file
    #
    #         @property
    #         def bar_pattern_full_frame_file(self) -> Path:
    #             img_file = self.ctx.workfolder / f'bar_pattern_full_frame_{self.ctx.metric}.png'
    #             return img_file
    #
    #     paths = ProjectPaths()
    #
    #     def main():
    #         get_data()
    #         make_fit()
    #         calc_stats()
    #         make_hist()
    #         make_bar()
    #         make_boxplot()
    #
    #     def get_data():
    #         print('\n\n====== Get Data ======')
    #         # data[self.proj][self.tiling][self.metric]
    #         data_file = paths.data_file
    #
    #         if paths.data_file.exists() and not overwrite:
    #             warning(f'\n  The data file "{ProjectPaths.data_file}" exist. Loading date.')
    #             return
    #
    #         data = AutoDict()
    #
    #         for self.proj in self.proj_list:
    #             for self.tiling in self.tiling_list:
    #                 for self.metric in ['time', 'time_std', 'rate']:
    #                     print(f'\r  Getting - {self.proj} {self.tiling} {self.metric}... ', end='')
    #
    #                     bulcket = data[self.proj][self.tiling][self.metric] = []
    #
    #                     for self.video in self.config.videos_list:
    #                         if self.proj not in self.video: continue
    #                         with self.dectime_ctx() as dectime:
    #                             for self.quality in self.quality_list:
    #                                 for self.chunk in self.chunk_list:
    #                                     # values["time"|"time_std"|"rate"]: list[float|int]
    #
    #                                     total = 0
    #                                     for self.tile in self.tile_list:
    #                                         values = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
    #                                         if self.metric == 'time':
    #                                             total += np.average(values['dectimes'])
    #                                         elif self.metric == 'rate':
    #                                             total += np.average(values['bitrate'])
    #                                     bulcket.append(total)
    #
    #         print(f' Saving... ', end='')
    #         save_json(data, paths.data_file)
    #         del (data)
    #         print(f'  Finished.')
    #
    #     def make_fit():
    #         print(f'\n\n====== Make Fit - Bins = {self.bins} ======')
    #
    #         # Load data file
    #         data = load_json(paths.data_file)
    #
    #         for self.proj in self.proj_list:
    #             for self.tiling in self.tiling_list:
    #                 for self.metric in self.metric_list:
    #                     print(f'  Fitting - {self.proj} {self.tiling} {self.metric}... ', end='')
    #
    #                     # Check fitter pickle
    #                     if paths.fitter_pickle_file.exists and not overwrite:
    #                         print(f'Pickle found! Skipping.')
    #                         continue
    #
    #                     # Load data file
    #                     samples = data[self.proj][self.tiling][self.metric]
    #
    #                     # Calculate bins
    #                     bins = self.bins
    #                     if self.bins == 'custom':
    #                         min_ = np.min(samples)
    #                         max_ = np.max(samples)
    #                         norm = round((max_ - min_) / 0.001)
    #                         if norm > 30:
    #                             bins = 30
    #                         else:
    #                             bins = norm
    #
    #                     # Make the fit
    #                     distributions = self.config['distributions']
    #                     fitter = Fitter(samples, bins=bins, distributions=distributions,
    #                                     timeout=900)
    #                     fitter.fit()
    #
    #                     # Saving
    #                     print(f'  Saving... ')
    #                     save_pickle(fitter, paths.fitter_pickle_file)
    #                     del (data)
    #                     print(f'  Finished.')
    #
    #     def calc_stats():
    #         print('  Calculating Statistics')
    #
    #         # Check stats file
    #         if paths.stats_file.exists() and not overwrite:
    #             print(f'  stats_file found! Skipping.')
    #             return
    #
    #         data = load_json(paths.data_file)
    #         stats = defaultdict(list)
    #
    #         for self.proj in self.proj_list:
    #             for self.tiling in self.tiling_list:
    #                 data_bucket = {}
    #
    #                 for self.metric in self.metric_list:
    #                     # Load data
    #                     data_bucket[self.metric] = data[self.proj][self.tiling][self.metric]
    #
    #                     # Load fitter pickle
    #                     fitter = load_pickle(ProjectPaths.fitter_pickle_file)
    #
    #                     # Calculate percentiles
    #                     percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
    #                     df_errors: pd.DataFrame = fitter.df_errors
    #
    #                     # Calculate errors
    #                     sse: pd.Series = df_errors['sumsquare_error']
    #                     bins = len(fitter.x)
    #                     rmse = np.sqrt(sse / bins)
    #                     nrmse = rmse / (sse.max() - sse.min())
    #
    #                     # Append info and stats on Dataframe
    #                     stats[f'proj'].append(self.proj)
    #                     stats[f'tiling'].append(self.tiling)
    #                     stats[f'metric'].append(self.metric)
    #                     stats[f'bins'].append(bins)
    #                     stats[f'average'].append(np.average(data_bucket[self.metric]))
    #                     stats[f'std'].append(float(np.std(data_bucket[self.metric])))
    #                     stats[f'min'].append(percentile[0])
    #                     stats[f'quartile1'].append(percentile[1])
    #                     stats[f'median'].append(percentile[2])
    #                     stats[f'quartile3'].append(percentile[3])
    #                     stats[f'max'].append(percentile[4])
    #
    #                     # Append distributions on Dataframe
    #                     for dist in sse.keys():
    #                         if dist not in fitter.fitted_param and dist == 'rayleigh':
    #                             fitter.fitted_param[dist] = (0., 0.)
    #                         params = fitter.fitted_param[dist]
    #                         dist_info = self.find_dist(dist, params)
    #
    #                         stats[f'rmse_{dist}'].append(rmse[dist])
    #                         stats[f'nrmse_{dist}'].append(nrmse[dist])
    #                         stats[f'sse_{dist}'].append(sse[dist])
    #                         stats[f'param_{dist}'].append(dist_info['parameters'])
    #                         stats[f'loc_{dist}'].append(dist_info['loc'])
    #                         stats[f'scale_{dist}'].append(dist_info['scale'])
    #
    #                 corr = np.corrcoef((data_bucket['time'], data_bucket['rate']))[1][0]
    #                 stats[f'correlation'].append(corr)  # for time
    #                 stats[f'correlation'].append(corr)  # for rate
    #
    #         pd.DataFrame(stats).to_csv(str(paths.stats_file), index=False)
    #
    #     def make_hist():
    #         print(f'\n====== Make Plot - Bins = {self.bins} ======')
    #         n_dist = 3
    #
    #         color_list = {'burr12': 'tab:blue', 'fatiguelife': 'tab:orange',
    #                       'gamma': 'tab:green', 'invgauss': 'tab:red',
    #                       'rayleigh': 'tab:purple', 'lognorm': 'tab:brown',
    #                       'genpareto': 'tab:pink', 'pareto': 'tab:gray',
    #                       'halfnorm': 'tab:olive', 'expon': 'tab:cyan'}
    #
    #         # Load data
    #         data = load_json(paths.data_file)
    #
    #         # make an image for each metric
    #         for self.metric in self.metric_list:
    #             # Check image file by metric
    #             if paths.hist_pattern_full_frame_file.exists() and not overwrite:
    #                 warning(f'Figure exist. Skipping')
    #                 continue
    #
    #             # Make figure
    #             fig = figure.Figure(figsize=(12.8, 3.84))
    #             pos = [(2, 5, x) for x in range(1, 5 * 2 + 1)]
    #             subplot_pos = iter(pos)
    #
    #             if self.metric == 'time':
    #                 xlabel = 'Decoding time (s)'
    #                 # scilimits = (-3, -3)
    #             else:
    #                 xlabel = 'Bit Rate (Mbps)'
    #                 # scilimits = (6, 6)
    #
    #             for self.proj in self.proj_list:
    #                 for self.tiling in self.tiling_list:
    #                     # Load fitter and select samples
    #                     fitter = load_pickle(ProjectPaths.fitter_pickle_file)
    #                     samples = data[self.proj][self.tiling][self.metric]
    #
    #                     # Position of plot
    #                     nrows, ncols, index = next(subplot_pos)
    #                     ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #
    #                     # Make the histogram
    #                     self.make_graph('hist', ax, y=samples, bins=len(fitter.x),
    #                                     label='empirical', title=f'{self.proj.upper()}-{self.tiling}',
    #                                     xlabel=xlabel)
    #
    #                     # make plot for n_dist distributions
    #                     dists = fitter.df_errors['sumsquare_error'].sort_values()[0:n_dist].index
    #                     for dist_name in dists:
    #                         fitted_pdf = fitter.fitted_pdf[dist_name]
    #                         self.make_graph('plot', ax, x=fitter.x, y=fitted_pdf,
    #                                         label=f'{dist_name}',
    #                                         color=color_list[dist_name])
    #
    #                     # ax.set_yscale('log')
    #                     ax.legend(loc='upper right')
    #
    #             print(f'  Saving the figure')
    #             fig.savefig(paths.hist_pattern_full_frame_file)
    #
    #     def make_bar():
    #         print(f'\n====== Make Bar - Bins = {self.bins} ======')
    #
    #         path = paths.bar_pattern_full_frame_file
    #         if path.exists() and not overwrite:
    #             warning(f'Figure exist. Skipping')
    #             return
    #
    #         stats = pd.read_csv(paths.stats_file)
    #         fig = figure.Figure(figsize=(6.4, 3.84))
    #         pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
    #         subplot_pos = iter(pos)
    #
    #         for self.metric in self.metric_list:
    #             nrows, ncols, index = next(subplot_pos)
    #             ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #
    #             for start, self.proj in enumerate(self.proj_list):
    #                 data = stats[(stats[f'proj'] == self.proj) & (stats['metric'] == self.metric)]
    #                 data_avg = data[f'average']
    #                 data_std = data[f'std']
    #
    #                 if self.metric == 'time':
    #                     ylabel = 'Decoding time (ms)'
    #                     scilimits = (-3, -3)
    #                     ax.set_ylim(0.230, 1.250)
    #                 else:
    #                     ylabel = 'Bit Rate (Mbps)'
    #                     scilimits = (6, 6)
    #
    #                 if self.proj == 'cmp':
    #                     color = 'tab:green'
    #                 else:
    #                     color = 'tab:blue'
    #
    #                 x = list(range(0 + start, len(data[f'tiling']) * 3 + start, 3))
    #
    #                 self.make_graph('bar', ax=ax,
    #                                 x=x, y=data_avg, yerr=data_std,
    #                                 color=color,
    #                                 ylabel=ylabel,
    #                                 title=f'{self.metric}',
    #                                 scilimits=scilimits)
    #
    #             # finishing of Graphs
    #             patch1 = mpatches.Patch(color='tab:green', label='CMP')
    #             patch2 = mpatches.Patch(color='tab:blue', label='ERP')
    #             legend = {'handles': (patch1, patch2), 'loc': 'upper right'}
    #             ax.legend(**legend)
    #             ax.set_xticks([i + 0.5 for i in range(0, len(self.tiling_list) * 3, 3)])
    #             ax.set_xticklabels(self.tiling_list)
    #
    #         print(f'Salvando a figura')
    #         fig.savefig(path)
    #
    #     def make_boxplot():
    #         print(f'\n====== Make Bar - Bins = {self.bins} ======')
    #
    #         path = paths.boxplot_pattern_full_frame_file
    #         if path.exists() and not overwrite:
    #             warning(f'Figure exist. Skipping')
    #             return
    #
    #         data = load_json(paths.data_file)
    #
    #         fig = figure.Figure(figsize=(6.4, 3.84))
    #         pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
    #         subplot_pos = iter(pos)
    #
    #         for self.metric in self.metric_list:
    #             nrows, ncols, index = next(subplot_pos)
    #             ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #
    #             for start, self.proj in enumerate(self.proj_list):
    #                 data_bucket = []
    #
    #                 for self.tiling in self.tiling_list:
    #                     data_bucket.append(data[self.proj][self.tiling][self.metric])
    #
    #                 if self.metric == 'time':
    #                     ylabel = 'Decoding time (ms)'
    #                     scilimits = (-3, -3)
    #                 else:
    #                     ylabel = 'Bit Rate (Mbps)'
    #                     scilimits = (6, 6)
    #
    #                 if self.proj == 'cmp':
    #                     color = 'tab:green'
    #                 else:
    #                     color = 'tab:blue'
    #
    #                 x = list(range(0 + start, len(self.tiling_list) * 3 + start, 3))
    #
    #                 self.make_graph('boxplot', ax=ax, x=x, y=data_bucket,
    #                                 title=f'{self.proj.upper()}-{self.metric}',
    #                                 ylabel=ylabel,
    #                                 scilimits=scilimits,
    #                                 color=color)
    #                 patch1 = mpatches.Patch(color=color, label='CMP')
    #                 patch2 = mpatches.Patch(color=color, label='ERP')
    #                 legend = {'handles': (patch1, patch2),
    #                           'loc': 'upper right'}
    #                 ax.legend(**legend)
    #                 ax.set_xticks([i + 0.5 for i in range(0, len(self.tiling_list) * 3, 3)])
    #                 ax.set_xticklabels(self.tiling_list)
    #
    #         print(f'Salvando a figura')
    #         fig.savefig(path)
    #
    #     main()

    # def by_video_by_tiling_by_quality(self, overwrite):
    #     class ProjectPaths:
    #         @staticmethod
    #         def data_file() -> Path:
    #             data_file = self.workfolder_data / f'data.json'
    #             return data_file
    #
    #         @staticmethod
    #         def fitter_pickle_file() -> Path:
    #             fitter_file = self.workfolder_data / f'fitter_{self.proj}_{self.video}_{self.tiling}_{self.quality}_{self.metric}_{self.bins}bins.pickle'
    #             return fitter_file
    #
    #         @staticmethod
    #         def stats_file() -> Path:
    #             stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
    #             return stats_file
    #
    #         @staticmethod
    #         def hist_video_quality_file() -> Path:
    #             img_file = self.workfolder / f'hist_video_pattern_quality_{self.video}_{self.quality}_{self.metric}_{self.bins}bins.png'
    #             return img_file
    #
    #         @staticmethod
    #         def bar_video_pattern_quality_file() -> Path:
    #             img_file = self.workfolder / f'bar_video_pattern_quality_{self.name}.png'
    #             return img_file
    #
    #         @staticmethod
    #         def bar_tiling_quality_video_file() -> Path:
    #             # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
    #             img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}_{self.metric}.png'
    #             return img_file
    #
    #         @staticmethod
    #         def boxplot_tiling_quality_video_file() -> Path:
    #             # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
    #             img_file = self.workfolder / f'boxplot_tiling_quality_video_{self.tiling}_{self.metric}.png'
    #             return img_file
    #
    #         @staticmethod
    #         def boxplot_quality_tiling_video_file() -> Path:
    #             # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
    #             img_file = self.workfolder / f'boxplot_quality_tiling_video_{self.quality}_{self.metric}.png'
    #             return img_file
    #
    #     def main():
    #         get_data()
    #         make_fit()
    #         calc_stats()
    #         # make_bar_video_tiling_quality()
    #         # make_bar_tiling_quality_video()
    #         # make_boxplot()
    #         make_boxplot2()
    #
    #     def get_data():
    #         print('\n\n====== Get Data ======')
    #
    #         if ProjectPaths.data_file().exists() and not overwrite:
    #             print(f'  The data file "{ProjectPaths.data_file()}" exist. Skipping... ', end='')
    #             return
    #         data = AutoDict()
    #
    #         for self.proj in self.proj_list:
    #             for self.video in self.videos_list:
    #                 if self.proj not in self.video: continue
    #                 for self.tiling in self.tiling_list:
    #                     for self.quality in self.quality_list:
    #                         for self.metric in ['time', 'time_std', 'rate']:
    #                             print(f'  Getting -  {self.proj} {self.name} {self.tiling} CRF{self.quality} {self.metric}... ', end='')
    #
    #                             bucket = data[self.name][self.proj][self.tiling][self.quality][self.metric] = []
    #
    #                             with self.dectime_ctx() as dectime:
    #                                 for self.tile in self.tile_list:
    #                                     for self.chunk in self.chunk_list:
    #                                         # values["time"|"time_std"|"rate"]: list[float|int]
    #                                         values = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
    #                                         bucket.append(np.average(values[self.metric]))
    #                             print(f'  OK.')
    #
    #         print(f' Saving... ', end='')
    #
    #         save_json(data, ProjectPaths.data_file())
    #         print(f'  Finished.')
    #
    #     def make_fit():
    #         print(f'\n\n====== Make Fit - Bins = {self.bins} ======')
    #
    #         # Load data file
    #         data = load_json(ProjectPaths.data_file())
    #
    #         for self.video in self.videos_list:
    #             for self.proj in self.proj_list:
    #                 if self.proj not in self.video: continue
    #                 for self.tiling in self.tiling_list:
    #                     for self.quality in self.quality_list:
    #                         for self.metric in self.metric_list:
    #                             print(f'  Fitting - {self.proj} {self.video} {self.tiling} CRF{self.quality} {self.metric}... ', end='')
    #
    #                             # Check fitter pickle
    #                             if ProjectPaths.fitter_pickle_file().exists() and not overwrite:
    #                                 print(f'Pickle found! Skipping.')
    #                                 continue
    #
    #                             # Calculate bins
    #                             bins = self.bins
    #                             if self.bins == 'custom':
    #                                 min_ = np.min(data)
    #                                 max_ = np.max(data)
    #                                 norm = round((max_ - min_) / 0.001)
    #                                 if norm > 30:
    #                                     bins = 30
    #                                 else:
    #                                     bins = norm
    #
    #                             # Make the fit
    #                             samples = data[self.name][self.proj][self.tiling][self.quality][self.metric]
    #                             distributions = self.config['distributions']
    #                             ft = Fitter(samples, bins=bins, distributions=distributions,
    #                                         timeout=900)
    #                             ft.fit()
    #
    #                             # Saving
    #                             print(f'  Saving... ', end='')
    #                             save_pickle(ft, ProjectPaths.fitter_pickle_file())
    #
    #                             print(f'  Finished.')
    #         del data
    #
    #     def calc_stats():
    #         print(f'\n\n====== Make Statistics - Bins = {self.bins} ======')
    #         # Check stats file
    #         if ProjectPaths.stats_file().exists() and not overwrite:
    #             print(f'  stats_file found! Skipping.')
    #             return
    #
    #         data = load_json(ProjectPaths.data_file())
    #         stats = defaultdict(list)
    #         for self.proj in self.proj_list:
    #             for self.video in self.videos_list:
    #                 if self.proj not in self.video: continue
    #                 for self.tiling in self.tiling_list:
    #                     for self.quality in self.quality_list:
    #                         data_bucket = {}
    #
    #                         for self.metric in self.metric_list:
    #                             # Load data file
    #                             data_bucket[self.metric] = data[self.name][self.proj][self.tiling][self.quality][self.metric]
    #
    #                             # Load fitter pickle
    #                             fitter = load_pickle(ProjectPaths.fitter_pickle_file())
    #
    #                             # Calculate percentiles
    #                             percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
    #                             df_errors: pd.DataFrame = fitter.df_errors
    #
    #                             # Calculate errors
    #                             sse: pd.Series = df_errors['sumsquare_error']
    #                             bins = len(fitter.x)
    #                             rmse = np.sqrt(sse / bins)
    #                             nrmse = rmse / (sse.max() - sse.min())
    #
    #                             # Append info and stats on Dataframe
    #                             stats[f'proj'].append(self.proj)
    #                             stats[f'video'].append(self.video)
    #                             stats[f'name'].append(self.name)
    #                             stats[f'tiling'].append(self.tiling)
    #                             stats[f'quality'].append(self.quality)
    #                             stats[f'metric'].append(self.metric)
    #                             stats[f'bins'].append(bins)
    #                             stats[f'average'].append(np.average(data_bucket[self.metric]))
    #                             stats[f'std'].append(float(np.std(data_bucket[self.metric])))
    #                             stats[f'min'].append(percentile[0])
    #                             stats[f'quartile1'].append(percentile[1])
    #                             stats[f'median'].append(percentile[2])
    #                             stats[f'quartile3'].append(percentile[3])
    #                             stats[f'max'].append(percentile[4])
    #
    #                             # Append distributions on Dataframe
    #                             for dist in sse.keys():
    #                                 if dist not in fitter.fitted_param and dist == 'rayleigh':
    #                                     fitter.fitted_param[dist] = (0., 0.)
    #                                 params = fitter.fitted_param[dist]
    #                                 dist_info = self.find_dist(dist, params)
    #
    #                                 stats[f'rmse_{dist}'].append(rmse[dist])
    #                                 stats[f'nrmse_{dist}'].append(nrmse[dist])
    #                                 stats[f'sse_{dist}'].append(sse[dist])
    #                                 stats[f'param_{dist}'].append(dist_info['parameters'])
    #                                 stats[f'loc_{dist}'].append(dist_info['loc'])
    #                                 stats[f'scale_{dist}'].append(dist_info['scale'])
    #
    #                         corr = np.corrcoef((data_bucket['time'], data_bucket['rate']))[1][0]
    #                         stats[f'correlation'].append(corr)  # for time
    #                         stats[f'correlation'].append(corr)  # for rate
    #
    #         pd.DataFrame(stats).to_csv(str(ProjectPaths.stats_file()), index=False)
    #
    #     def make_bar_video_tiling_quality():
    #         print(f'\n====== Make Bar1 - Bins = {self.bins} ======')
    #         stats = pd.read_csv(ProjectPaths.stats_file())
    #
    #         for self.video in stats['name'].unique():
    #             if ProjectPaths.bar_video_pattern_quality_file().exists() and not overwrite:
    #                 warning(f'Figure exist. Skipping')
    #                 return
    #
    #             fig = figure.Figure()
    #             subplot_pos = iter(((2, 5, 1), (2, 5, 2), (2, 5, 3), (2, 5, 4), (2, 5, 5), (2, 5, 6), (2, 5, 7), (2, 5, 8), (2, 5, 9), (2, 5, 10)))
    #             for self.proj in self.proj_list:
    #                 for self.tiling in self.tiling_list:
    #                     nrows, ncols, index = next(subplot_pos)
    #                     ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #                     time_stats = stats[(stats['name'] == self.video) & (stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'time')]
    #                     x = xticks = time_stats['quality']
    #                     time_avg = time_stats['average']
    #                     time_std = time_stats['std']
    #                     ylabel = 'Decoding time (s)' if index in [1, 6] else None
    #
    #                     self.make_graph('bar', ax=ax,
    #                                     x=x, y=time_avg, yerr=time_std,
    #                                     title=f'{self.proj}-{self.tiling}',
    #                                     xlabel='CRF',
    #                                     ylabel=ylabel,
    #                                     xticks=xticks,
    #                                     width=5,
    #                                     scilimits=(-3, -3))
    #
    #                     # line plot of bit rate
    #                     ax = ax.twinx()
    #
    #                     rate_stats = stats[(stats['name'] == self.video) & (stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'rate')]
    #                     rate_avg = rate_stats['average']
    #                     rate_stdp = rate_avg - rate_stats['std']
    #                     rate_stdm = rate_avg + rate_stats['std']
    #                     legend = {'handles': (mpatches.Patch(color='#1f77b4', label='Time'),
    #                                           mlines.Line2D([], [], color='red', label='Bitrate')),
    #                               'loc': 'upper right'}
    #                     ylabel = 'Bit Rate (Mbps)' if index in [5, 10] else None
    #
    #                     self.make_graph('plot', ax=ax,
    #                                     x=x, y=rate_avg,
    #                                     legend=legend,
    #                                     ylabel=ylabel,
    #                                     color='r',
    #                                     scilimits=(6, 6))
    #
    #                     ax.plot(x, rate_stdp, color='gray', linewidth=1)
    #                     ax.plot(x, rate_stdm, color='gray', linewidth=1)
    #                     ax.ticklabel_format(axis='y', style='scientific',
    #                                         scilimits=(6, 6))
    #             # fig.show()
    #             print(f'Salvando a figura')
    #
    #             fig.savefig(ProjectPaths.bar_video_pattern_quality_file())
    #
    #     def make_bar_tiling_quality_video():
    #         print(f'\n====== Make Bar2 - Bins = {self.bins} ======')
    #         stats_df = pd.read_csv(ProjectPaths.stats_file())
    #
    #         for self.tiling in self.tiling_list:
    #             # pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
    #             # subplot_pos = iter(pos)
    #             # fig = figure.Figure(figsize=(12, 4), dpi=300)
    #
    #             for self.metric in ['time', 'rate']:
    #                 if ProjectPaths.bar_tiling_quality_video_file().exists() and not overwrite:
    #                     warning(f'Figure exist. Skipping')
    #                     return
    #
    #                 pos = [(1, 1, 1)]
    #                 subplot_pos = iter(pos)
    #                 fig = figure.Figure(figsize=(12, 4), dpi=300)
    #
    #                 nrows, ncols, index = next(subplot_pos)
    #                 ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #                 xticks = stats_df['name'].unique()
    #                 start = -1
    #                 for self.quality in self.quality_list:
    #                     for self.proj in self.proj_list:
    #                         start += 1
    #                         stats = stats_df[(stats_df['quality'] == int(self.quality)) & (stats_df['tiling'] == self.tiling) & (stats_df['proj'] == self.proj) & (
    #                                     stats_df['metric'] == self.metric)]
    #                         stats = stats.sort_values(by=['average'], ascending=False)
    #
    #                         x = list(range(0 + start, len(xticks) * 13 + start, 13))
    #                         time_avg = stats['average']
    #                         time_std = stats['std']
    #
    #                         if self.metric == 'time':
    #                             ylabel = 'Decoding time (s)'
    #                             scilimits = (-3, -3)
    #                         else:
    #                             ylabel = 'Bit Rate (Mbps)'
    #                             scilimits = (6, 6)
    #
    #                         if self.proj == 'cmp':
    #                             color = 'lime'
    #                         else:
    #                             color = 'blue'
    #                         self.make_graph('bar', ax=ax,
    #                                         x=x, y=time_avg, yerr=time_std,
    #                                         title=f'{self.tiling}-{self.metric}',
    #                                         ylabel=ylabel,
    #                                         width=1,
    #                                         color=color,
    #                                         scilimits=scilimits)
    #                 ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
    #                 ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
    #                 print(f'Salvando a figura')
    #                 fig.savefig(ProjectPaths.bar_tiling_quality_video_file())
    #
    #             # print(f'Salvando a figura')
    #             # fig.savefig(ProjectPaths.bar_tiling_quality_video_file())
    #
    #     def make_boxplot():
    #         print(f'\n====== Make Boxplot - Bins = {self.bins} ======')
    #         stats_df = pd.read_csv(ProjectPaths.stats_file())
    #         data = load_json(ProjectPaths.data_file())
    #
    #         for self.tiling in self.tiling_list:
    #             for self.metric in self.metric_list:
    #                 if ProjectPaths.boxplot_tiling_quality_video_file().exists() and not overwrite:
    #                     warning(f'Figure exist. Skipping')
    #                     return
    #
    #                 stats_df = stats_df.sort_values(['average'], ascending=False)
    #
    #                 pos = [(1, 1, 1)]
    #                 subplot_pos = iter(pos)
    #                 fig = figure.Figure(figsize=(20, 8), dpi=300)
    #
    #                 nrows, ncols, index = next(subplot_pos)
    #                 ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #                 ax.grid(axis='y')
    #                 xticks = stats_df['name'].unique()
    #                 start = -1
    #
    #                 for self.quality in self.quality_list:
    #                     for self.proj in ['cmp', 'erp']:
    #                         start += 1
    #                         data_bucket = []
    #
    #                         for name in xticks:
    #                             data_bucket.append(data[name][self.proj][self.tiling][self.quality][self.metric])
    #                         x = list(range(0 + start, len(xticks) * 13 + start, 13))
    #
    #                         if self.metric == 'time':
    #                             ylabel = 'Decoding time (s)'
    #                             scilimits = (-3, -3)
    #                         else:
    #                             ylabel = 'Bit Rate (Mbps)'
    #                             scilimits = (6, 6)
    #
    #                         if self.proj == 'cmp':
    #                             color = 'lime'
    #                         else:
    #                             color = 'blue'
    #
    #                         self.make_graph('boxplot', ax=ax,
    #                                         x=x, y=data_bucket,
    #                                         title=f'{self.tiling}-{self.metric}',
    #                                         ylabel=ylabel,
    #                                         width=1,
    #                                         color=color,
    #                                         scilimits=scilimits)
    #                         patch1 = mpatches.Patch(color='lime', label='CMP')
    #                         patch2 = mpatches.Patch(color='blue', label='ERP')
    #                         legend = {'handles': (patch1, patch2),
    #                                   'loc': 'upper right'}
    #                         ax.legend(**legend)
    #
    #                 ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
    #                 ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
    #                 print(f'Salvando a figura')
    #                 fig.savefig(ProjectPaths.boxplot_tiling_quality_video_file())
    #
    #     def make_boxplot2():
    #         print(f'\n====== Make Boxplot2 - Bins = {self.bins} ======')
    #         stats_df = pd.read_csv(ProjectPaths.stats_file())
    #         data = load_json(ProjectPaths.data_file())
    #
    #         for self.quality in self.quality_list:
    #             # pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
    #             # subplot_pos = iter(pos)
    #             # fig = figure.Figure(figsize=(12, 4), dpi=300)
    #
    #             for self.metric in ['time', 'rate']:
    #                 if ProjectPaths.boxplot_quality_tiling_video_file().exists() and not overwrite:
    #                     warning(f'Figure exist. Skipping')
    #                     return
    #
    #                 stats_df = stats_df.sort_values(['average'], ascending=False)
    #
    #                 pos = [(1, 1, 1)]
    #                 subplot_pos = iter(pos)
    #                 fig = figure.Figure(figsize=(20, 8), dpi=300)
    #
    #                 nrows, ncols, index = next(subplot_pos)
    #                 ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
    #                 ax.grid(axis='y')
    #                 xticks = stats_df['name'].unique()
    #                 start = -1
    #                 for self.tiling in self.tiling_list:
    #                     for self.proj in ['cmp', 'erp']:
    #                         start += 1
    #                         data_bucket = []
    #                         for name in xticks:
    #                             data_bucket.append(data[name][self.proj][self.tiling][self.quality][self.metric])
    #                         x = list(range(0 + start, len(xticks) * 13 + start, 13))
    #
    #                         if self.metric == 'time':
    #                             ylabel = 'Decoding time (s)'
    #                             scilimits = (-3, -3)
    #                         else:
    #                             ylabel = 'Bit Rate (Mbps)'
    #                             scilimits = (6, 6)
    #
    #                         if self.proj == 'cmp':
    #                             color = 'lime'
    #                         else:
    #                             color = 'blue'
    #
    #                         self.make_graph('boxplot', ax=ax,
    #                                         x=x, y=data_bucket,
    #                                         title=f'CRF{self.quality}-{self.metric}',
    #                                         ylabel=ylabel,
    #                                         width=1,
    #                                         color=color,
    #                                         scilimits=scilimits)
    #                         patch1 = mpatches.Patch(color='lime', label='CMP')
    #                         patch2 = mpatches.Patch(color='blue', label='ERP')
    #                         legend = {'handles': (patch1, patch2),
    #                                   'loc': 'upper right'}
    #                         ax.legend(**legend)
    #
    #                 ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
    #                 ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
    #                 print(f'Salvando a figura')
    #                 fig.savefig(ProjectPaths.boxplot_quality_tiling_video_file())
    #
    #     main()

class ____ByUser____: ...

class GetTilesPath(Worker, GlobalPaths):
    @property
    def workfolder(self) -> Path:
        folder = self.project_path / self.graphs_folder / f'{self.__class__.__name__}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def dataset_json(self) -> Path:
        self.database_folder = Path('datasets') / self.config['dataset_name']
        return self.database_folder / f'{self.config["dataset_name"]}.json'

    @property
    def get_tiles_path(self) -> Path:
        folder = self.project_path / self.get_tiles_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def get_tiles_json(self) -> Path:
        path = self.get_tiles_path / f'get_tiles_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}.json'
        return path


class UserDectimeOptions(Enum):
    USERS_METRICS = 0
    VIEWPORT_METRICS = 1
    GET_TILES = 2


class UserDectime:
    def __init__(self, config: str, role: UserDectimeOptions):
        operations = {UserDectimeOptions(0): self.UsersMetrics,
                      UserDectimeOptions(1): self.ViewportMetrics,
                      UserDectimeOptions(2): self.GetTiles,
                      }
        operations[role](Config(config))
        print(f'\n====== The end of {role} ======')

    class GetTiles(GetTilesPath):
        results: AutoDict
        database: AutoDict
        old_video = None
        old_tiling = None
        old_chunk = None
        new_database: AutoDict
        get_tiles: AutoDict
        name: str

        def __init__(self, config: Config):
            self.config = config
            self.dataset_name = config['dataset_name']
            self.print_resume()
            # self.process_nasrabadi(overwrite=False)
            self.get_tiles(overwrite=False)
            # self.user_analisys(overwrite=False)

        def process_nasrabadi(self, overwrite=False):
            print('Processing dataset.')
            database_json = self.dataset_json


            if overwrite or not database_json.exists():
                pi = np.pi
                pi2 = np.pi * 2
                rotation_map = {'cable_cam_nas': 265 / 180 * pi, 'drop_tower_nas': 180 / 180 * pi,
                                'wingsuit_dubai_nas': 63 / 180 * pi, 'drone_chases_car_nas': 81 / 180 * pi}
                database = AutoDict()
                user_map = {}
                video_id_map = load_json(f'{self.dataset_json.parent}/videos_map.json', object_hook=dict)

                def process_vectors(frame_counter, previous_line, actual_line):
                    timestamp, x, y, z = actual_line
                    frame_timestamp = frame_counter / 30

                    if timestamp < frame_timestamp:
                        raise ValueError
                    elif timestamp > frame_timestamp:
                        # Linear Interpolation
                        old_timestamp, old_x, old_y, old_z = previous_line
                        x, y, z = lin_interpol(frame_timestamp, (timestamp, old_timestamp), ((x, y, z), (old_x, old_y, old_z)))

                    yaw, pitch = map(round, cart2hcs(x, y, z), (6, 6))
                    roll = 0

                    if video_name in rotation_map:
                        yaw -= rotation_map[video_name]
                    yaw = (yaw + pi) % pi2 - pi
                    return yaw, pitch, roll

                for csv_database_file in self.dataset_json.parent.glob('*/*.csv'):
                    # Parse filename in user_id and video_id
                    user_nas_id, video_nas_id = csv_database_file.stem.split('_')
                    video_name = video_id_map[video_nas_id]

                    # Map user_id to str(int)
                    try:
                        user_id = user_map[user_nas_id]
                    except KeyError:
                        user_map[user_nas_id] = str(len(user_map))
                        user_id = user_map[user_nas_id]

                    print(f'\rUser {user_id} - {video_name} - ', end='')

                    # "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
                    # respectively, to reorient during playback." - Author
                    # Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
                    # rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0
                    previous_line = None
                    yaw_pitch_roll_frames = []
                    start_time = time.time()
                    n = 0
                    frame_counter = 0
                    names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
                    head_movement = pd.read_csv(csv_database_file, names=names)


                    for n, line in enumerate(head_movement.itertuples(index=False, name=None)):
                        timestamp, Qx, Qy, Qz, Qw, Vx, Vy, Vz = map(float, line)
                        x, y, z = Vz, Vy, Vx  # Based on gitHub code of author
                        try:
                            yaw, pitch, roll = process_vectors(frame_counter, previous_line, (timestamp, x, y, z))
                            yaw_pitch_roll_frames.append((yaw, pitch, roll))
                            frame_counter += 1
                            if frame_counter >= 1800: break
                        except ValueError:
                            pass
                        previous_line = timestamp, x, y, z

                    while len(yaw_pitch_roll_frames) < 1800:
                        yaw_pitch_roll_frames.append(yaw_pitch_roll_frames[-1])
                    database[video_name][user_id] = yaw_pitch_roll_frames
                    print(f'Samples {n:04d} - {frame_counter=} - {time.time() - start_time:0.3f}')

                print(f'\nFinish. Saving as {database_json}.')
                save_json(database, database_json)
            else:
                warning(f'The file {database_json} exist. Skipping.')

        def get_tiles(self, overwrite=False):
            print(f'Get Tiles.')
            dataset_json = None
            erp_list = {self.tiling: ERP(self.tiling, '576x288', '110x90') for self.tiling in self.tiling_list}
            for self.video in self.videos_list:
                if self.get_tiles_json.exists() and not overwrite:
                    warning(f'\nThe file {self.get_tiles_json} exist. Loading.')
                    continue

                try:
                    users_data = dataset_json[self.name]
                except (TypeError, NameError):
                    dataset_json = load_json(self.dataset_json)
                    users_data = dataset_json[self.name]


                results = AutoDict()

                for self.tiling in self.tiling_list:
                    erp = erp_list[self.tiling]

                    for user in users_data:
                        print(f'{self.name} - tiling {self.tiling} - User {user}')
                        if self.tiling == '1x1':
                            get_tiles_value = {'frame': [0]*1800, 'chunks': [0]*60}
                            results[self.vid_proj][self.name][self.tiling][user] = get_tiles_value
                            continue

                        get_tiles_value = defaultdict(list)
                        results[self.vid_proj][self.name][self.tiling][user] = get_tiles_value
                        result_frames = []
                        result_chunks = {}
                        chunk = 0
                        tiles_chunks = set()

                        start = time.time()
                        for frame, (yaw, pitch, roll) in enumerate(users_data[user]):
                            if (frame + 1) % 30 == 0:
                                print(f'\r  {user} - {frame:04d} - ', end='')

                            # vptiles
                            erp.rotate(*np.deg2rad((yaw, pitch, roll)))
                            vptiles: list[int] = erp.get_vptiles()
                            result_frames.append(vptiles)

                            # Calcule vptiles by chunk
                            tiles_chunks.update(vptiles)
                            if (frame + 1) % 30 == 0:
                                chunk += 1  # start from 1 gpac defined
                                result_chunks[f'{chunk}'] = list(tiles_chunks)
                                tiles_chunks = set()
                                print(f'{time.time() - start:.3f}s - {tiles_chunks}          ', end='')

                        print('')
                        get_tiles_value['frame'].append(result_frames)
                        get_tiles_value['chunks'].append(result_chunks)
                    exit(0)
                        # todo: remove this in the future
                        # break
                print(f'Saving {self.get_tiles_json}')
                save_json(results, self.get_tiles_json)

        def user_analisys(self, overwrite=False):
            counter_tiles_json = self.get_tiles_path / f'counter_tiles_{self.dataset_name}.json'

            if counter_tiles_json.exists():
                result = load_json(counter_tiles_json)
            else:
                database = load_json(self.database_json, object_hook=dict)
                result = {}
                for self.tiling in self.tiling_list:
                    # Colect tiling count
                    tiles_counter = Counter()
                    print(f'{self.tiling=}')
                    nb_chunk = 0
                    for self.video in self.videos_list:
                        users = database[self.name].keys()
                        get_tiles_json = self.get_tiles_path / f'get_tiles_{self.dataset_name}_{self.video}_{self.tiling}.json'
                        if not get_tiles_json.exists():
                            print(dict(tiles_counter))
                            break
                        print(f'  - {self.video=}')
                        get_tiles = load_json(get_tiles_json, object_hook=dict)
                        for user in users:
                            # hm = database[self.name][user]
                            chunks = get_tiles[self.vid_proj][self.tiling][user]['chunks'].keys()
                            for chunk in chunks:
                                seen_tiles_by_chunk = get_tiles[self.vid_proj][self.tiling][user]['chunks'][chunk]
                                tiles_counter = tiles_counter + Counter(seen_tiles_by_chunk)
                                nb_chunk += 1

                    # normalize results
                    dict_tiles_counter = dict(tiles_counter)
                    column = []
                    for tile_id in range(len(dict_tiles_counter)):
                        if not tile_id in dict_tiles_counter:
                            column.append(0.)
                        else:
                            column.append(dict_tiles_counter[tile_id] / nb_chunk)
                    result[self.tiling] = column
                    print(result)

                save_json(result, counter_tiles_json)
            # Create heatmap
            for self.tiling in self.tiling_list:
                tiling_result = result[self.tiling]
                shape = splitx(self.tiling)[::-1]
                grade = np.asarray(tiling_result).reshape(shape)
                fig, ax = plt.subplots()
                im = ax.imshow(grade, cmap='jet', )
                ax.set_title(f'Tiling {self.tiling}')
                fig.colorbar(im, ax=ax, label='chunk frequency')
                heatmap_tiling = self.get_tiles_path / f'heatmap_tiling_{self.dataset_name}_{self.tiling}.png'
                fig.savefig(f'{heatmap_tiling}')
                fig.show()

    class UsersMetrics(Worker, DectimeGraphsPaths):
        fit: Fitter
        fitter = AutoDict()
        fit_errors = AutoDict()
        data: Union[dict, AutoDict] = None
        color_list = {'burr12': 'tab:blue',
                      'fatiguelife': 'tab:orange',
                      'gamma': 'tab:green',
                      'invgauss': 'tab:red',
                      'rayleigh': 'tab:purple',
                      'lognorm': 'tab:brown',
                      'genpareto': 'tab:pink',
                      'pareto': 'tab:gray',
                      'halfnorm': 'tab:olive',
                      'expon': 'tab:cyan'}
        stats: dict[str, list] = None
        seen_tiles_data = None

        def __init__(self, config):
            self.config = config
            self.print_resume()
            self.n_dist = 6
            self.bins = 30
            self.stats = defaultdict(list)
            self.corretations_bucket = defaultdict(list)

            self.get_data_bucket()
            self.make_fit()
            self.calc_stats()

            DectimeGraphs.rc_config()
            self.make_hist(overwrite=True)
            # self.make_boxplot(overwrite=True)
            # self.make_violinplot(overwrite=True)

        def get_get_tiles_data(self, overwrite=False):
            print('\n====== Get tiles data ======')
            self.seen_tiles_data = AutoDict()

            if self.seen_tiles_data_file.exists() and not overwrite:
                warning(f'  The data file "{self.seen_tiles_data_file}" exist. Loading date.')
                return

            for self.video in self.videos_list:
                time_data = load_json(self.dectime_result_json, object_hook=dict)
                rate_data = load_json(self.bitrate_result_json, object_hook=dict)
                qlt_data = load_json(self.quality_result_json, object_hook=dict)

                for self.tiling in self.tiling_list:
                    get_tiles_data = load_json(self.get_tiles_result_json, object_hook=dict)
                    users = get_tiles_data[self.vid_proj][self.tiling].keys()
                    print(f'\r  Get Tiles - {self.vid_proj}  {self.name} {self.tiling} - {len(users)} users ... ', end='')

                    for user in users:
                        for self.chunk in self.chunk_list:
                            get_tiles_val = get_tiles_data[self.vid_proj][self.tiling][user]['chunks'][self.chunk]
                            for self.quality in self.quality_list:
                                for self.tile in self.tile_list:
                                    if int(self.tile) in get_tiles_val:
                                        dectime_val = time_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                        bitrate_val = rate_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                        quality_val = qlt_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']

                                        seen_tiles_result = self.seen_tiles_data[f'{user}'][self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                        seen_tiles_result['time'] = float(np.average(dectime_val['times']))
                                        seen_tiles_result['rate'] = float(bitrate_val['rate'])
                                        seen_tiles_result['time_std'] = float(np.std(dectime_val['times']))

                                        for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                            value = quality_val[metric]
                                            if value == float('inf'):
                                                value = 1000
                                            seen_tiles_result[metric] = value
                    print('OK')

            print(f'  Saving get tiles... ', end='')
            save_json(self.seen_tiles_data, self.seen_tiles_data_file)
            print(f'  Finished.')

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            # Check file
            if not overwrite and self.data_bucket_file.exists():
                print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                return

            data_bucket = AutoDict()
            json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                           'time': self.dectime_result_json,
                                           'time_std': self.dectime_result_json,
                                           'PSNR': self.quality_result_json,
                                           'WS-PSNR': self.quality_result_json,
                                           'S-PSNR': self.quality_result_json}[metric]
            def bucket():
                # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
                data = data_bucket[self.metric][self.vid_proj][self.tiling]
                if not isinstance(data, list):
                    data = data_bucket[self.metric][self.vid_proj][self.tiling] = []
                return data

            def process(value):
                # Process value according the metric
                if self.metric == 'time':
                    new_value = float(np.average(value['times']))
                elif self.metric == 'time_std':
                    new_value = float(np.std(value['times']))
                elif self.metric == 'rate':
                    new_value = float(value['rate'])
                else:
                    # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                    metric_value = value[self.metric]
                    new_value = metric_value
                    # value = metric_value if float(metric_value) != float('inf') else 1000
                return new_value

            for self.metric in self.metric_list:
                for self.video in self.videos_list:
                    data =  load_json(json_metrics(self.metric), object_hook=dict)
                    for self.tiling in self.tiling_list:
                        tiling_data = data[self.vid_proj][self.tiling]
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')
                        for self.quality in self.quality_list:
                            qlt_data = tiling_data[self.quality]
                            for self.tile in self.tile_list:
                                tile_data = qlt_data[self.tile]
                                for self.chunk in self.chunk_list:
                                    chunk_data = tile_data[self.chunk]
                                    bucket().append(process(chunk_data))

            if remove_outliers: self.remove_outliers(data_bucket)

            print(f'  Saving  {self.metric}... ', end='')
            save_json(data_bucket, self.data_bucket_file)
            print(f'  Finished.')

        def make_fit(self, overwrite=False):
            print(f'\n\n====== Making Fit - Bins = {self.bins} ======')
            data_bucket = None
            distributions = self.config['distributions']

            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        print(f'  Fitting - {self.metric} {self.proj} {self.tiling}... ', end='')

                        if not overwrite and self.fitter_pickle_file.exists():
                            # Check fitter pickle
                            print(f'Pickle found! Skipping.')
                            continue

                        try:
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                            samples = data_bucket[self.metric][self.proj][self.tiling]

                        # Make a fit
                        fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                        fitter.fit()

                        # Save
                        print(f'  Saving... ', end='')
                        save_pickle(fitter, self.fitter_pickle_file)
                        print(f'  Finished.')

        def calc_stats(self, overwrite=False):
            print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
            data_bucket = None

            if overwrite or not self.stats_file.exists():
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        for self.metric in self.metric_list:
                            # Get samples and Fitter
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                            fitter = load_pickle(self.fitter_pickle_file)

                            # Calculate percentiles
                            percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                            # Calculate errors
                            df_errors: pd.DataFrame = fitter.df_errors
                            sse: pd.Series = df_errors['sumsquare_error']
                            bins = len(fitter.x)
                            rmse = np.sqrt(sse / bins)
                            nrmse = rmse / (sse.max() - sse.min())

                            # Append info and stats on Dataframe
                            self.stats[f'proj'].append(self.proj)
                            self.stats[f'tiling'].append(self.tiling)
                            self.stats[f'metric'].append(self.metric)
                            self.stats[f'bins'].append(self.bins)

                            self.stats[f'average'].append(np.average(samples))
                            self.stats[f'std'].append(float(np.std(samples)))

                            self.stats[f'min'].append(percentile[0])
                            self.stats[f'quartile1'].append(percentile[1])
                            self.stats[f'median'].append(percentile[2])
                            self.stats[f'quartile3'].append(percentile[3])
                            self.stats[f'max'].append(percentile[4])

                            # Append distributions on Dataframe
                            for dist in sse.keys():
                                params = fitter.fitted_param[dist]
                                dist_info = DectimeGraphs.find_dist(dist, params)

                                self.stats[f'rmse_{dist}'].append(rmse[dist])
                                self.stats[f'nrmse_{dist}'].append(nrmse[dist])
                                self.stats[f'sse_{dist}'].append(sse[dist])
                                self.stats[f'param_{dist}'].append(dist_info['parameters'])
                                self.stats[f'loc_{dist}'].append(dist_info['loc'])
                                self.stats[f'scale_{dist}'].append(dist_info['scale'])

                pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
            else:
                print(f'  stats_file found! Skipping.')

            if overwrite or not self.correlations_file.exists():
                if not data_bucket:
                    data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                from itertools import combinations
                corretations_bucket = defaultdict(list)

                for metric1, metric2 in combinations(self.metric_list, r=2):
                    self.metric = metric1
                    data_bucket1 = load_json(self.data_bucket_file, object_hook=dict)
                    self.metric = metric2
                    data_bucket2 = load_json(self.data_bucket_file, object_hook=dict)

                    for self.proj in self.proj_list:
                        for self.tiling in self.tiling_list:
                            samples1 = data_bucket1[self.proj][self.tiling]
                            samples2 = data_bucket2[self.proj][self.tiling]
                            corrcoef = np.corrcoef((samples1, samples2))[1][0]

                            corretations_bucket[f'proj'].append(self.proj)
                            corretations_bucket[f'tiling'].append(self.tiling)
                            corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                            corretations_bucket[f'corr'].append(corrcoef)

                pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)
            else:
                print(f'  correlations_file found! Skipping.')

        def make_hist(self, overwrite=False):
            print(f'\n====== Make Histogram - Bins = {self.bins} ======')
            folder = self.workfolder / 'pdf_cdf'
            folder.mkdir(parents=True, exist_ok=True)

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    im_file = folder / f'pdf_{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if im_file.exists() and not overwrite:
                        warning(f'Histogram exist. Skipping')
                        continue

                    fig_pdf: figure.Figure = plt.figure(figsize=(12.0, 2))  # pdf
                    fig_cdf: figure.Figure = plt.Figure(figsize=(12.0, 2))  # cdf
                    subplot_pos = [(1, 5, x) for x in range(1, 5 * 1 + 1)]  # 1x5

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Load fitter
                        fitter = load_pickle(self.fitter_pickle_file)
                        dists = fitter.df_errors['sumsquare_error'].sort_values()[0:self.n_dist].index

                        # <editor-fold desc="Make PDF">
                        # Create a subplot
                        ax: axes.Axes = fig_cdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        ax.bar(fitter.x, fitter.y, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # Make plot for n_dist distributions
                        for dist_name in dists:
                            fitted_pdf = fitter.fitted_pdf[dist_name]
                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'

                            ax.plot(fitter.x, fitted_pdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Density' if index in [1, 6] else None
                        legkwrd = {'loc': 'upper right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                        # <editor-fold desc="Make CDF">
                        ax: axes.Axes = fig_pdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        bins_height = np.cumsum([y * (fitter.x[1] - fitter.x[0]) for y in fitter.y])
                        ax.bar(fitter.x, bins_height, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # make plot for n_dist distributions
                        for dist_name in dists:
                            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                            param = fitter.fitted_param[dist_name]
                            fitted_cdf = dist.cdf(fitter.x, *param)

                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'
                            ax.plot(fitter.x, fitted_cdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Cumulative' if index in [1, 6] else None
                        legkwrd = {'loc': 'lower right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                    print(f'  Saving the PDF')
                    fig_pdf.savefig(im_file)

                    print(f'  Saving the CDF ')
                    im_file = folder / f'cdf_{self.metric}_{self.proj}.png'
                    fig_cdf.savefig(im_file)

        def make_boxplot(self, overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')
            folder = self.workfolder / 'boxplot'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]
            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file)
                for self.proj in self.proj_list:
                    img_file = folder / f'boxplot_pattern_{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if img_file.exists() and not overwrite:
                        warning(f'BoxPlot exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        suptitle = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        suptitle = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = None
                        suptitle = self.metric
                    # </editor-fold>

                    fig_boxplot = plt.Figure(figsize=(6., 2.))
                    fig_boxplot.suptitle(f'{suptitle}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        tiling_data = data_bucket[self.proj][self.tiling]

                        # if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                        #     tiling_data = [data for data in tiling_data if data < 1000]

                        ax: axes.Axes = fig_boxplot.add_subplot(nrows, ncols, index)
                        boxplot_sep = ax.boxplot((tiling_data,), widths=0.8,
                                                 whis=(0, 100),
                                                 showfliers=False,
                                                 boxprops=dict(facecolor='tab:blue'),
                                                 flierprops=dict(color='r'),
                                                 medianprops=dict(color='k'),
                                                 patch_artist=True)
                        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                        ax.set_xticks([0])
                        ax.set_xticklabels([self.tiling_list[index - 1]])
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # if index in [columns]:
                        #     ax.legend(handles=legend_handles, loc='upper left',
                        #                   bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig_boxplot.savefig(img_file)

        def make_violinplot(self, overwrite=False):
            print(f'\n====== Make Violin - Bins = {self.bins} ======')
            folder = self.workfolder / 'violinplot'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file)
                for self.proj in self.proj_list:
                    img_file = folder / f'violinplot_pattern_{self.metric}_{self.proj}.png'

                    if img_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        title = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        title = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        title = self.metric
                    # </editor-fold>

                    fig = figure.Figure(figsize=(6.8, 3.84))
                    fig.suptitle(f'{title}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        tiling_data = data_bucket[self.proj][self.tiling]

                        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            tiling_data = [data for data in tiling_data if data < 1000]

                        ax_sep: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax_sep.violinplot([tiling_data], positions=[1],
                                          showmedians=True, widths=0.9)

                        ax_sep.set_xticks([1])
                        # ax_sep.set_ylim(bottom=0)
                        ax_sep.set_xticklabels([self.tiling_list[index-1]])
                        ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # if index in [columns]:
                        #     ax_sep.legend(handles=legend_handles, loc='upper left',
                        #                   bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    class ViewportMetrics(Worker, TileDecodeBenchmarkPaths, DectimeGraphsPaths):
        fit: Fitter
        fitter = AutoDict()
        fit_errors = AutoDict()
        data: Union[dict, AutoDict] = None
        color_list = {'burr12': 'tab:blue',
                      'fatiguelife': 'tab:orange',
                      'gamma': 'tab:green',
                      'invgauss': 'tab:red',
                      'rayleigh': 'tab:purple',
                      'lognorm': 'tab:brown',
                      'genpareto': 'tab:pink',
                      'pareto': 'tab:gray',
                      'halfnorm': 'tab:olive',
                      'expon': 'tab:cyan'}
        stats: dict[str, list] = None
        seen_tiles_data = None

        def __init__(self, config):
            self.config = config
            self.print_resume()
            self.make_quality()

        def make_quality(self, overwrite=False):
            database = load_json(self.dataset_json)

            for self.video in self.videos_list:
                frame_height, frame_width, n_channels = self.video_shape
                nfov = NFOV(frame_width, frame_height)
                pw, ph = splitx(self.resolution)
                users_data = database[self.name]
                img_proj_ref = np.zeros(self.video_shape, dtype='uint8')
                # img_proj = np.zeros(self.video_shape, dtype='uint8')

                for self.tiling in self.tiling_list:
                    if self.tiling == '1x1': continue
                    get_tiles_data = load_json(self.get_tiles_result_json, object_hook=dict)
                    users_list = get_tiles_data[self.vid_proj][self.tiling].keys()

                    result = AutoDict()
                    folder = self.project_path / 'user_quality'
                    folder.mkdir(parents=True, exist_ok=True)
                    result_name =  folder / f'user_quality_{self.config["dataset_name"]}_{self.video}_{self.tiling}.json'

                    if overwrite or not result_name.exists():
                        M, N = splitx(self.tiling)
                        tw, th = int(pw / M), int(ph / N)

                        for user in users_list:
                            for self.quality in self.quality_list:
                                video_ref_name = folder / f"seen_user{user}_{self.vid_proj}_{self.name}_{self.tiling}_CRF0.mp4"
                                video_writer = skvideo.io.FFmpegWriter(video_ref_name,
                                                                       inputdict={'-r': '30'},
                                                                       outputdict={'-crf': '0','-r': '30', '-pix_fmt': 'yuv420p'})
                                yaw_pitch_roll_frames = iter(users_data[user])

                                for self.chunk in self.chunk_list:
                                    get_tiles_val: list[int] = get_tiles_data[self.vid_proj][self.tiling][user]['chunks'][self.chunk]

                                    # tiles_reader: dict[str, skvideo.io.FFmpegReader] = {str(self.tile): skvideo.io.FFmpegReader(f'{self.segment_file}').nextFrame() for self.tile in get_tiles_val}
                                    tiles_reader_ref: dict[str, skvideo.io.FFmpegReader] = {str(self.tile): skvideo.io.FFmpegReader(f'{self.reference_segment}').nextFrame() for self.tile in get_tiles_val}

                                    # img_proj *= 0
                                    img_proj_ref *= 0

                                    psnr_chunk: list[float] = []
                                    mse_chunk: list[float] = []
                                    for _ in range(30):  # 30 frames per chunk
                                        # Build projection frame
                                        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2), dpi=200)
                                        start = time.time()
                                        for self.tile in map(str, get_tiles_val):
                                            tile_x, tile_y = idx2xy(int(self.tile), (N, M))
                                            tx, ty = tile_x * tw, tile_y * th

                                            # tile_frame = next(tiles_reader[self.tile])
                                            tile_frame_ref = next(tiles_reader_ref[self.tile])

                                            # img_proj[ty:ty + th, tx:tx + tw, :] = tile_frame
                                            img_proj_ref[ty:ty + th, tx:tx + tw, :] = tile_frame_ref

                                        print(f'time to mount {time.time() - start: 0.3f} s')
                                        yaw, pitch, roll = next(yaw_pitch_roll_frames)
                                        center_point = np.array([np.deg2rad(yaw), np.deg2rad(pitch)])  # camera center point (valid range [0,1])
                                        # fov_video = nfov.toNFOV(img_proj, center_point).astype('float64')
                                        start = time.time()
                                        fov_ref = nfov.get_viewport(img_proj_ref, center_point)  #.astype('float64')
                                        print(f'time to extract vp =  {time.time() - start: 0.3f} s')

                                        # mse = np.mean((fov_ref - fov_video) ** 2)
                                        # psnr = 20 * np.log10(255. / np.sqrt(mse))
                                        #
                                        # mse_chunk.append(float(mse))
                                        # psnr_chunk.append(float(psnr))
                                        ax[0].imshow(nfov.frame)
                                        ax[0].axis('off')
                                        ax[1].imshow(fov_ref)
                                        ax[1].axis('off')
                                        plt.tight_layout()
                                        plt.show()
                                        video_writer.writeFrame(fov_ref)
                                        plt.close()

                                    video_writer.close()
                                    exit()
                                    # result[self.vid_proj][self.name][self.tiling][self.quality][self.chunk][user]['mse'] = np.mean(mse_chunk)
                                    # result[self.vid_proj][self.name][self.tiling][self.quality][self.chunk][user]['psnr'] = np.mean(psnr_chunk)
                                print('')
                                video_writer.close()
                        save_json(result, result_name)
                    else:
                        print(f'File {result_name} exists. skipping')

        def get_get_tiles_data(self, overwrite=False):
            print('\n====== Get tiles data ======')
            self.seen_tiles_data = AutoDict()

            if self.seen_tiles_data_file.exists() and not overwrite:
                warning(f'  The data file "{self.seen_tiles_data_file}" exist. Loading date.')
                return

            for self.video in self.videos_list:
                time_data = load_json(self.dectime_result_json, object_hook=dict)
                rate_data = load_json(self.bitrate_result_json, object_hook=dict)
                qlt_data = load_json(self.quality_result_json, object_hook=dict)

                for self.tiling in self.tiling_list:
                    get_tiles_data = load_json(self.get_tiles_result_json, object_hook=dict)
                    users = get_tiles_data[self.vid_proj][self.tiling].keys()
                    print(f'\r  Get Tiles - {self.vid_proj}  {self.name} {self.tiling} - {len(users)} users ... ', end='')

                    for user in users:
                        for self.chunk in self.chunk_list:
                            get_tiles_val = get_tiles_data[self.vid_proj][self.tiling][user]['chunks'][self.chunk]
                            for self.quality in self.quality_list:
                                for self.tile in self.tile_list:
                                    if int(self.tile) in get_tiles_val:
                                        dectime_val = time_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                        bitrate_val = rate_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                        quality_val = qlt_data[self.vid_proj][self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']

                                        seen_tiles_result = self.seen_tiles_data[f'{user}'][self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
                                        seen_tiles_result['time'] = float(np.average(dectime_val['times']))
                                        seen_tiles_result['rate'] = float(bitrate_val['rate'])
                                        seen_tiles_result['time_std'] = float(np.std(dectime_val['times']))

                                        for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                            value = quality_val[metric]
                                            if value == float('inf'):
                                                value = 1000
                                            seen_tiles_result[metric] = value
                    print('OK')

            print(f'  Saving get tiles... ', end='')
            save_json(self.seen_tiles_data, self.seen_tiles_data_file)
            print(f'  Finished.')

        def get_data_bucket(self, overwrite=False, remove_outliers=False):
            print(f'\n\n====== Getting Data - Bins = {self.bins} ======')

            # Check file
            if not overwrite and self.data_bucket_file.exists():
                print(f'  The data file "{self.data_bucket_file}" exist. Skipping.')
                return

            data_bucket = AutoDict()
            json_metrics = lambda metric: {'rate': self.bitrate_result_json,
                                           'time': self.dectime_result_json,
                                           'time_std': self.dectime_result_json,
                                           'PSNR': self.quality_result_json,
                                           'WS-PSNR': self.quality_result_json,
                                           'S-PSNR': self.quality_result_json}[metric]
            def bucket():
                # [metric][vid_proj][tiling] = [video, quality, tile, chunk]
                data = data_bucket[self.metric][self.vid_proj][self.tiling]
                if not isinstance(data, list):
                    data = data_bucket[self.metric][self.vid_proj][self.tiling] = []
                return data

            def process(value):
                # Process value according the metric
                if self.metric == 'time':
                    new_value = float(np.average(value['times']))
                elif self.metric == 'time_std':
                    new_value = float(np.std(value['times']))
                elif self.metric == 'rate':
                    new_value = float(value['rate'])
                else:
                    # if self.metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                    metric_value = value[self.metric]
                    new_value = metric_value
                    # value = metric_value if float(metric_value) != float('inf') else 1000
                return new_value

            for self.metric in self.metric_list:
                for self.video in self.videos_list:
                    data =  load_json(json_metrics(self.metric), object_hook=dict)
                    for self.tiling in self.tiling_list:
                        tiling_data = data[self.vid_proj][self.tiling]
                        print(f'\r  {self.metric} - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')
                        for self.quality in self.quality_list:
                            qlt_data = tiling_data[self.quality]
                            for self.tile in self.tile_list:
                                tile_data = qlt_data[self.tile]
                                for self.chunk in self.chunk_list:
                                    chunk_data = tile_data[self.chunk]
                                    bucket().append(process(chunk_data))

            if remove_outliers: self.remove_outliers(data_bucket)

            print(f'  Saving  {self.metric}... ', end='')
            save_json(data_bucket, self.data_bucket_file)
            print(f'  Finished.')

        def make_fit(self, overwrite=False):
            print(f'\n\n====== Making Fit - Bins = {self.bins} ======')
            data_bucket = None
            distributions = self.config['distributions']

            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        print(f'  Fitting - {self.metric} {self.proj} {self.tiling}... ', end='')

                        if not overwrite and self.fitter_pickle_file.exists():
                            # Check fitter pickle
                            print(f'Pickle found! Skipping.')
                            continue

                        try:
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                        except TypeError:
                            data_bucket = load_json(self.data_bucket_file, object_hook=dict)
                            samples = data_bucket[self.metric][self.proj][self.tiling]

                        # Make a fit
                        fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                        fitter.fit()

                        # Save
                        print(f'  Saving... ', end='')
                        save_pickle(fitter, self.fitter_pickle_file)
                        print(f'  Finished.')

        def calc_stats(self, overwrite=False):
            print(f'\n\n====== Making Statistics - Bins = {self.bins} ======')
            data_bucket = None

            if overwrite or not self.stats_file.exists():
                data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        for self.metric in self.metric_list:
                            # Get samples and Fitter
                            samples = data_bucket[self.metric][self.proj][self.tiling]
                            fitter = load_pickle(self.fitter_pickle_file)

                            # Calculate percentiles
                            percentile = np.percentile(samples, [0, 25, 50, 75, 100]).T

                            # Calculate errors
                            df_errors: pd.DataFrame = fitter.df_errors
                            sse: pd.Series = df_errors['sumsquare_error']
                            bins = len(fitter.x)
                            rmse = np.sqrt(sse / bins)
                            nrmse = rmse / (sse.max() - sse.min())

                            # Append info and stats on Dataframe
                            self.stats[f'proj'].append(self.proj)
                            self.stats[f'tiling'].append(self.tiling)
                            self.stats[f'metric'].append(self.metric)
                            self.stats[f'bins'].append(self.bins)

                            self.stats[f'average'].append(np.average(samples))
                            self.stats[f'std'].append(float(np.std(samples)))

                            self.stats[f'min'].append(percentile[0])
                            self.stats[f'quartile1'].append(percentile[1])
                            self.stats[f'median'].append(percentile[2])
                            self.stats[f'quartile3'].append(percentile[3])
                            self.stats[f'max'].append(percentile[4])

                            # Append distributions on Dataframe
                            for dist in sse.keys():
                                params = fitter.fitted_param[dist]
                                dist_info = DectimeGraphs.find_dist(dist, params)

                                self.stats[f'rmse_{dist}'].append(rmse[dist])
                                self.stats[f'nrmse_{dist}'].append(nrmse[dist])
                                self.stats[f'sse_{dist}'].append(sse[dist])
                                self.stats[f'param_{dist}'].append(dist_info['parameters'])
                                self.stats[f'loc_{dist}'].append(dist_info['loc'])
                                self.stats[f'scale_{dist}'].append(dist_info['scale'])

                pd.DataFrame(self.stats).to_csv(self.stats_file, index=False)
            else:
                print(f'  stats_file found! Skipping.')

            if overwrite or not self.correlations_file.exists():
                if not data_bucket:
                    data_bucket = load_json(self.data_bucket_file, object_hook=dict)

                from itertools import combinations
                corretations_bucket = defaultdict(list)

                for metric1, metric2 in combinations(self.metric_list, r=2):
                    self.metric = metric1
                    data_bucket1 = load_json(self.data_bucket_file, object_hook=dict)
                    self.metric = metric2
                    data_bucket2 = load_json(self.data_bucket_file, object_hook=dict)

                    for self.proj in self.proj_list:
                        for self.tiling in self.tiling_list:
                            samples1 = data_bucket1[self.proj][self.tiling]
                            samples2 = data_bucket2[self.proj][self.tiling]
                            corrcoef = np.corrcoef((samples1, samples2))[1][0]

                            corretations_bucket[f'proj'].append(self.proj)
                            corretations_bucket[f'tiling'].append(self.tiling)
                            corretations_bucket[f'metric'].append(f'{metric1}_{metric2}')
                            corretations_bucket[f'corr'].append(corrcoef)

                pd.DataFrame(corretations_bucket).to_csv(self.correlations_file, index=False)
            else:
                print(f'  correlations_file found! Skipping.')

        def make_hist(self, overwrite=False):
            print(f'\n====== Make Histogram - Bins = {self.bins} ======')
            folder = self.workfolder / 'pdf_cdf'
            folder.mkdir(parents=True, exist_ok=True)

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    im_file = folder / f'pdf_{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if im_file.exists() and not overwrite:
                        warning(f'Histogram exist. Skipping')
                        continue

                    fig_pdf: figure.Figure = plt.figure(figsize=(12.0, 2))  # pdf
                    fig_cdf: figure.Figure = plt.Figure(figsize=(12.0, 2))  # cdf
                    subplot_pos = [(1, 5, x) for x in range(1, 5 * 1 + 1)]  # 1x5

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Load fitter
                        fitter = load_pickle(self.fitter_pickle_file)
                        dists = fitter.df_errors['sumsquare_error'].sort_values()[0:self.n_dist].index

                        # <editor-fold desc="Make PDF">
                        # Create a subplot
                        ax: axes.Axes = fig_cdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        ax.bar(fitter.x, fitter.y, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # Make plot for n_dist distributions
                        for dist_name in dists:
                            fitted_pdf = fitter.fitted_pdf[dist_name]
                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'

                            ax.plot(fitter.x, fitted_pdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Density' if index in [1, 6] else None
                        legkwrd = {'loc': 'upper right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                        # <editor-fold desc="Make CDF">
                        ax: axes.Axes = fig_pdf.add_subplot(nrows, ncols, index)

                        # Make bars of histogram
                        bins_height = np.cumsum([y * (fitter.x[1] - fitter.x[0]) for y in fitter.y])
                        ax.bar(fitter.x, bins_height, label='empirical', color='#dbdbdb',
                               width=fitter.x[1] - fitter.x[0])

                        # make plot for n_dist distributions
                        for dist_name in dists:
                            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                            param = fitter.fitted_param[dist_name]
                            fitted_cdf = dist.cdf(fitter.x, *param)

                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'
                            ax.plot(fitter.x, fitted_cdf,
                                    color=self.color_list[dist_name], label=label)

                        # <editor-fold desc="Format plot">
                        title = f'{self.proj.upper()}-{self.tiling}'
                        ylabel = 'Cumulative' if index in [1, 6] else None
                        legkwrd = {'loc': 'lower right'}

                        if self.metric == 'time':
                            scilimits = (-3, -3)
                            xlabel = f'Average Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'time_std':
                            scilimits = (-3, -3)
                            xlabel = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                        elif self.metric == 'rate':
                            scilimits = (6, 6)
                            xlabel = f'Bit {self.metric.capitalize()} (Mbps)'
                        else:
                            scilimits = (0, 0)
                            xlabel = self.metric
                        # </editor-fold>

                        ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax.set_title(title)
                        ax.set_xlabel(xlabel)
                        ax.set_ylabel(ylabel)
                        # ax.set_yscale('log')
                        ax.legend(**legkwrd)
                        # </editor-fold>

                    print(f'  Saving the PDF')
                    fig_pdf.savefig(im_file)

                    print(f'  Saving the CDF ')
                    im_file = folder / f'cdf_{self.metric}_{self.proj}.png'
                    fig_cdf.savefig(im_file)

        def make_boxplot(self, overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')
            folder = self.workfolder / 'boxplot'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]
            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file)
                for self.proj in self.proj_list:
                    img_file = folder / f'boxplot_pattern_{self.metric}_{self.proj}.png'

                    # Check image file by metric
                    if img_file.exists() and not overwrite:
                        warning(f'BoxPlot exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        suptitle = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        suptitle = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        suptitle = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = None
                        suptitle = self.metric
                    # </editor-fold>

                    fig_boxplot = plt.Figure(figsize=(6., 2.))
                    fig_boxplot.suptitle(f'{suptitle}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        tiling_data = data_bucket[self.proj][self.tiling]

                        # if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                        #     tiling_data = [data for data in tiling_data if data < 1000]

                        ax: axes.Axes = fig_boxplot.add_subplot(nrows, ncols, index)
                        boxplot_sep = ax.boxplot((tiling_data,), widths=0.8,
                                                 whis=(0, 100),
                                                 showfliers=False,
                                                 boxprops=dict(facecolor='tab:blue'),
                                                 flierprops=dict(color='r'),
                                                 medianprops=dict(color='k'),
                                                 patch_artist=True)
                        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                        ax.set_xticks([0])
                        ax.set_xticklabels([self.tiling_list[index - 1]])
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # if index in [columns]:
                        #     ax.legend(handles=legend_handles, loc='upper left',
                        #                   bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig_boxplot.savefig(img_file)

        def make_violinplot(self, overwrite=False):
            print(f'\n====== Make Violin - Bins = {self.bins} ======')
            folder = self.workfolder / 'violinplot'
            folder.mkdir(parents=True, exist_ok=True)

            row, columns = 1, 5
            subplot_pos = [(row, columns, x) for x in range(1, columns * row + 1)]
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['erp'], label='ERP'),
                              # mpatches.Patch(color=colors['cmp'], label='CMP'),
                              ]

            # make an image for each metric and projection
            for self.metric in self.metric_list:
                data_bucket = load_json(self.data_bucket_file)
                for self.proj in self.proj_list:
                    img_file = folder / f'violinplot_pattern_{self.metric}_{self.proj}.png'

                    if img_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # <editor-fold desc="Format plot">
                    if self.metric == 'time':
                        scilimits = (-3, -3)
                        title = f'Average Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'time_std':
                        scilimits = (-3, -3)
                        title = f'Std Dev - Decoding {self.metric.capitalize()} (ms)'
                    elif self.metric == 'rate':
                        scilimits = (6, 6)
                        title = f'Bit {self.metric.capitalize()} (Mbps)'
                    else:
                        scilimits = (0, 0)
                        title = self.metric
                    # </editor-fold>

                    fig = figure.Figure(figsize=(6.8, 3.84))
                    fig.suptitle(f'{title}')

                    for self.tiling, (nrows, ncols, index) in zip(self.tiling_list, subplot_pos):
                        # Get data
                        tiling_data = data_bucket[self.proj][self.tiling]

                        if self.metric in [ 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            tiling_data = [data for data in tiling_data if data < 1000]

                        ax_sep: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        ax_sep.violinplot([tiling_data], positions=[1],
                                          showmedians=True, widths=0.9)

                        ax_sep.set_xticks([1])
                        # ax_sep.set_ylim(bottom=0)
                        ax_sep.set_xticklabels([self.tiling_list[index-1]])
                        ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits)
                        # if index in [columns]:
                        #     ax_sep.legend(handles=legend_handles, loc='upper left',
                        #                   bbox_to_anchor=(1.01, 1.0), fontsize='small')

                    print(f'  Saving the figure')
                    fig.savefig(img_file)

        @staticmethod
        def remove_outliers(data, resumenamecsv=None):
            ### Fliers analysis data[self.proj][self.tiling][self.metric]
            print(f' Removing outliers... ', end='')
            resume = defaultdict(list)
            for proj in data:
                for tiling in data[proj]:
                    for metric in data[proj][tiling]:
                        data_bucket = data[proj][tiling][metric]

                        min, q1, med, q3, max = np.percentile(data_bucket, [0, 25, 50, 75, 100]).T
                        iqr = 1.5 * (q3 - q1)
                        clean_left = q1 - iqr
                        clean_right = q3 + iqr

                        data_bucket_clean = [d for d in data_bucket
                                             if (clean_left <= d <= clean_right)]
                        data[proj][tiling][metric] = data_bucket

                        resume['projection'] += [proj]
                        resume['tiling'] += [tiling]
                        resume['metric'] += [metric]
                        resume['min'] += [min]
                        resume['q1'] += [q1]
                        resume['median'] += [med]
                        resume['q3'] += [q3]
                        resume['max'] += [max]
                        resume['iqr'] += [iqr]
                        resume['clean_left'] += [clean_left]
                        resume['clean_right'] += [clean_right]
                        resume['original_len'] += [len(data_bucket)]
                        resume['clean_len'] += [len(data_bucket_clean)]
                        resume['clear_rate'] += [len(data_bucket_clean) / len(data_bucket)]
            print(f'  Finished.')
            if resumenamecsv is not None:
                pd.DataFrame(resume).to_csv(resumenamecsv)

    @staticmethod
    def rc_config():
        import matplotlib as mpl
        rc_param = {"figure": {'figsize': (7.0, 1.2), 'dpi': 600, 'autolayout': True},
                    "axes": {'linewidth': 0.5, 'titlesize': 8, 'labelsize': 6,
                             'prop_cycle': cycler(color=[plt.get_cmap('tab20')(i) for i in range(20)])},
                    "xtick": {'labelsize': 6},
                    "ytick": {'labelsize': 6},
                    "legend": {'fontsize': 6},
                    "font": {'size': 6},
                    "patch": {'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': '#3297c9'},
                    "lines": {'linewidth': 0.5, 'markersize': 2},
                    "errorbar": {'capsize': 4},
                    "boxplot": {'flierprops.marker': '+', 'flierprops.markersize': 1, 'flierprops.linewidth': 0.5,
                                'boxprops.linewidth': 0.0,
                                'capprops.linewidth': 1,
                                'medianprops.linewidth': 0.5,
                                'whiskerprops.linewidth': 0.5,
                                }
                    }

        for group in rc_param:
            mpl.rc(group, **rc_param[group])




class ____Others____: ...

class Dashing:
    def __init__(self, config: str = None, role: str = None, **kwargs):
        """

        # :param config:
        # :param role: Someone from operations dict
        # :param kwargs: Role parameters
        # """
        self.prepare = TileDecodeBenchmark.prepare
        self.compress = TileDecodeBenchmark.compress

        operations = {
            'PREPARE': Role(name='PREPARE', deep=1, init=None,
                            operation=self.prepare, finish=None),
            'COMPRESS': Role(name='COMPRESS', deep=4, init=None,
                             operation=self.compress, finish=None),
            'DASH': Role(name='SEGMENT', deep=4, init=None,
                         operation=self.dash, finish=None),
            'MEASURE_CHUNKS': Role(name='DECODE', deep=5, init=None,
                                   operation=self.measure, finish=None),
        }

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    def dash(self): pass

    def measure(self): pass


class CheckTiles:
    def __init__(self, config: str, role: str, **kwargs):
        operations = {
            'CHECK_ORIGINAL': Role(name='CHECK_ORIGINAL', deep=1, init=None,
                                   operation=self.check_original,
                                   finish=self.save),
            'CHECK_LOSSLESS': Role(name='CHECK_LOSSLESS', deep=1, init=None,
                                   operation=self.check_lossless,
                                   finish=self.save),
            'CHECK_COMPRESS': Role(name='CHECK_COMPRESS', deep=4, init=None,
                                   operation=self.check_compress,
                                   finish=self.save),
            'CHECK_SEGMENT': Role(name='CHECK_SEGMENT', deep=5, init=None,
                                  operation=self.check_segment,
                                  finish=self.save),
            'CHECK_DECODE': Role(name='CHECK_DECODE', deep=5, init=None,
                                 operation=self.check_decode,
                                 finish=self.save),
            'CHECK_RESULTS': Role(name='CHECK_RESULTS', deep=5,
                                  init=self.load_results,
                                  operation=self.check_dectime,
                                  finish=self.save),
            'CHECK_GET_TILES': Role(name='CHECK_GET_TILES', deep=2,
                                    init=self.init_check_get_tiles,
                                    operation=self.check_get_tiles,
                                    finish=None),
        }

        self.role = operations[role]
        self.check_table = {'file': [], 'msg': []}

        self.results = AutoDict()
        self.results_dataframe = pd.DataFrame()
        self.config = Config(config)
        self.video_context = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    def check_original(self, **check_video_kwargs):
        original_file = self.video_context.original_file
        debug(f'==== Checking {original_file} ====')
        msg = self.check_video(original_file, **check_video_kwargs)

        self.check_table['file'].append(original_file)
        self.check_table['msg'].append(msg)

    def check_lossless(self, **check_video_kwargs):
        lossless_file = self.video_context.lossless_file
        debug(f'Checking the file {lossless_file}')

        duration = self.video_context.video.duration
        fps = self.video_context.video.fps
        log_pattern = f'frame={duration * fps:5}'

        msg = self.check_video(lossless_file, log_pattern, **check_video_kwargs)

        self.check_table['file'].append(lossless_file)
        self.check_table['msg'].append(msg)

    def check_compress(self, only_error=True, **check_video_kwargs):
        video_file = self.video_context.compressed_file
        debug(f'Checking the file {video_file}')

        duration = self.video_context.video.duration
        fps = self.video_context.video.fps
        log_pattern = f'encoded {duration * fps} frames'

        msg = self.check_video(video_file, log_pattern, **check_video_kwargs)

        if not (only_error and msg == 'log_ok-video_ok'):
            self.check_table['file'].append(video_file)
            self.check_table['msg'].append(msg)

    def check_segment(self, only_error=True, **kwargs):
        segment_file = self.video_context.segment_file
        debug(f'Checking the file {segment_file}')

        msg = self.check_video(segment_file, **kwargs)

        if not (only_error and msg == 'log_ok-video_ok'):
            self.check_table['file'].append(segment_file)
            self.check_table['msg'].append(msg)

    def check_video(self, video: Path, log_pattern=None, check_log=False,
                    check_video=False, check_gop=False, clean=False,
                    deep_check=False) -> str:
        debug(f'Checking video {video}.')
        log = video.with_suffix('.log')
        msg = ['log_ok', 'video_ok']

        if check_log and log_pattern is not None:
            if not log.exists():
                msg[0] = 'log_not_found'
            elif log.stat().st_size == 0:
                msg[0] = 'log_size==0'
            else:
                log_content = log.read_text().splitlines()
                log_check_pattern = len(['' for line in log_content
                                         if log_pattern in line])
                if log_check_pattern == 0:
                    no_such_file = len(['' for line in log_content
                                        if 'No such file or directory'
                                        in line])
                    if no_such_file > 0:
                        msg[0] = 'log_vid_n_found'
                    else:
                        msg[0] = 'log_corrupt'

        if check_video:
            if not video.exists():
                msg[1] = 'video_not_found'
            elif video.stat().st_size == 0:
                msg[1] = 'video_size==0'
            else:
                if deep_check:
                    cmd = f'ffprobe -hide_banner -i {video}'
                    proc = run(cmd, shell=True, stderr=DEVNULL)
                    if proc.returncode != 0:
                        msg[1] = 'video_corrupt'
                if check_gop:
                    max_gop, gop = check_video_gop(video)
                    if max_gop != self.video_context.video.gop:
                        msg[1] = f'video_wrong_gop_={max_gop}'

        not_ok = 'video_ok' not in msg and 'log_ok' not in msg
        if not_ok and clean:
            warning(f'Cleaning {video}')
            msg[0] = msg[0] + '_clean'
            video.unlink(missing_ok=True)
            msg[1] = msg[1] + '_clean'
            log.unlink(missing_ok=True)
        return '-'.join(msg)

    @staticmethod
    def count_decoding(dectime_log) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :return:
        """
        try:
            content = dectime_log.read_text(encoding='utf-8').splitlines()
        except UnicodeDecodeError:
            warning('ERROR: UnicodeDecodeError. Cleaning.')
            dectime_log.unlink()
            return 0
        except FileNotFoundError:
            warning('ERROR: FileNotFoundError. Return 0.')
            return 0

        return len(['' for line in content if 'utime' in line])

    def check_decode(self, only_error=True, clean=False):
        dectime_log = self.video_context.dectime_log
        debug(f'Checking the file {dectime_log}')

        if not dectime_log.exists():
            warning('logfile_not_found')
            count_decode = 0
        else:
            count_decode = self.count_decoding(dectime_log)

        if count_decode == 0 and clean:
            dectime_log.unlink(missing_ok=True)

        if not (only_error and count_decode >= self.video_context.decoding_num):
            msg = f'decoded_{count_decode}x'
            self.check_table['file'].append(dectime_log)
            self.check_table['msg'].append(msg)

    def load_results(self):
        dectime_json_file = self.video_context.dectime_json_file
        self.results = load_json(dectime_json_file)

    def check_dectime(self, only_error=True):
        results = self.results
        ctx = self.video_context
        results = results[str(ctx.video)][str(ctx.tiling)][str(ctx.quality)][str(ctx.tile)][str(ctx.chunk)]

        if not (results == {}):
            bitrate = float(results['bitrate'])
            if bitrate > 0:
                msg = 'bitrate_ok'
            else:
                msg = 'bitrate==0'

            dectimes = results['dectimes']
            if len(dectimes) >= self.video_context.decoding_num:
                msg += '_dectimes_ok'
            else:
                msg += '_dectimes==0'

        else:
            warning(f'The result key for {self.video_context} is empty.')
            msg = 'empty_key'

        if not (only_error and msg == 'bitrate_ok_dectimes_ok'):
            key = self.video_context.make_name()
            self.check_table['file'].append(key)
            self.check_table['msg'].append(msg)

    def save(self):
        # Create Paths
        date = datetime.datetime.today()
        table_filename = f'{self.role.name}-table-{date}.csv'
        resume_filename = f'{self.role.name}-resume-{date}.csv'
        check_folder = self.video_context.check_folder
        table_filename = check_folder / table_filename.replace(':', '-')
        resume_filename = check_folder / resume_filename.replace(':', '-')

        # Write check table
        check_table = pd.DataFrame(self.check_table)
        check_table_csv = check_table.to_csv(index_label='counter')
        table_filename.write_text(check_table_csv, encoding='utf-8')

        # Create and Display Resume
        resume = dict(Counter(check_table['msg']))
        print('Resume:')
        print(json.dumps(resume, indent=2))

        # Write Resume
        resume_pd = pd.DataFrame.from_dict(resume, orient='index',
                                           columns=('count',))
        resume_pd_csv = resume_pd.to_csv(index_label='msg')
        resume_filename.write_text(resume_pd_csv, encoding='utf-8')

    def init_check_get_tiles(self):
        self.get_tiles_path = (self.video_context.project_path
                               / self.video_context.get_tiles_folder)

    def check_get_tiles(self, dataset_name):
        tiling = self.video_context.tiling
        if str(tiling) == '1x1':
            # info(f'skipping tiling 1x1')
            return

        video_name = self.video_context.video.name.replace("_cmp", "").replace("_erp", "")
        tiling.fov = '90x90'

        filename = f'get_tiles_{dataset_name}_{video_name}_{self.video_context.video.projection}_{tiling}.pickle'
        get_tiles_pickle = self.get_tiles_path / filename

        if not get_tiles_pickle.exists():
            warning(f'The file {get_tiles_pickle} NOT exist. Skipping.')
            return

        results = load_pickle(get_tiles_pickle)
        if len(results) != 30:
            print(f'{video_name} - tiling {tiling}: {len(results)} users')
            # for user_id in results_cmp:
            #     new_results[user_id] = results_erp[user_id]
            # get_tiles_pickle2 = self.get_tiles_path / f'get_tiles_{dataset_name}_{video_name}_{self.video_context.video.projection}_{tiling}-2.pickle'
            # # results2 = load_pickle(get_tiles_pickle2)
            # get_tiles_pickle.rename(get_tiles_pickle2)

            # save_pickle(new_results, get_tiles_pickle)

        for user_id in results:
            # if len(results[user_id]) > 1800:
            #     results[user_id] = results[user_id][:1800]
            # if len(results[user_id]) < 1800:
            #     while len(results[user_id]) < 1800:
            #         results[user_id].append(results[user_id][-1])
            if len(results[user_id]) != 1800:
                print(f'    {video_name} - tiling {tiling}: User {user_id}: {len(results[user_id])} samples')

        # get_tiles_pickle.rename(get_tiles_pickle2)
        get_tiles_pickle_new = self.get_tiles_path / f'get_tiles_{dataset_name}_{video_name}_{self.video_context.video.projection}.pickle'
        if get_tiles_pickle_new.exists():
            new_results = load_pickle(get_tiles_pickle_new)
        else:
            new_results = {}

        new_results[f'{tiling}'] = results
        save_pickle(new_results, get_tiles_pickle_new)


# <editor-fold desc="Siti2D">
# class Siti2D(BaseTileDecodeBenchmark):
#     def __init__(self, config: str = None, role: str = None, **kwargs):
#         """
#
#         :param config:
#         :param role: Someone from Role dict
#         :param kwargs: Role parameters
#         """
#
#         operations = {
#             'SITI': Role(name='PREPARE', deep=4, init=None,
#                          operation=self.siti, finish=self.end_siti),
#         }
#
#         self.config = Config(config)
#         self.config['tiling_list'] = ['1x1']
#         self.role = operations[role]
#         self.video_context = VideoContext(self.config, self.role.deep)
#         self.config['quality_list'] = [28]
#
#         self.run(**kwargs)
#
#     # 'CALCULATE SITI'
#     def siti(self, overwrite=False, animate_graph=False, save=True):
#         if not self.video_context.compressed_file.exists():
#             if not self.video_context.lossless_file.exists():
#                 warning(f'The file {self.video_context.lossless_file} not exist. '
#                         f'Skipping.')
#                 return 'skip'
#             self.compress()
#
#         siti = SiTi(self.video_context)
#         siti.calc_siti(animate_graph=animate_graph, overwrite=overwrite,
#                        save=save)
#
#     def compress(self):
#         compressed_file = self.video_context.compressed_file
#         compressed_log = self.video_context.compressed_file.with_suffix('.log')
#
#         debug(f'==== Processing {compressed_file} ====')
#
#         quality = self.video_context.quality
#         gop = self.video_context.gop
#         tile = self.video_context.tile
#
#         cmd = ['ffmpeg -hide_banner -y -psnr']
#         cmd += [f'-i {self.video_context.lossless_file}']
#         cmd += [f'-crf {quality} -tune "psnr"']
#         cmd += [f'-c:v libx265']
#         cmd += [f'-x265-params']
#         cmd += [f'"keyint={gop}:'
#                 f'min-keyint={gop}:'
#                 f'open-gop=0:'
#                 f'scenecut=0:'
#                 f'info=0"']
#         cmd += [f'-vf "crop='
#                 f'w={tile.resolution.W}:h={tile.resolution.H}:'
#                 f'x={tile.position.x}:y={tile.position.y}"']
#         cmd += [f'{compressed_file}']
#         cmd = ' '.join(cmd)
#
#         run_command(cmd, compressed_log, 'w')
#
#     def end_siti(self):
#         self._join_siti()
#         self._scatter_plot_siti()
#
#     def _join_siti(self):
#         siti_results_final = pd.DataFrame()
#         siti_stats_json_final = {}
#         num_frames = None
#
#         for name in enumerate(self.video_context.names_list):
#             """Join siti_results"""
#             siti_results_file = self.video_context.siti_results
#             siti_results_df = pd.read_csv(siti_results_file)
#             if num_frames is None:
#                 num_frames = self.video_context.duration * self.video_context.fps
#             elif num_frames < len(siti_results_df['si']):
#                 dif = len(siti_results_df['si']) - num_frames
#                 for _ in range(dif):
#                     siti_results_df.loc[len(siti_results_df)] = [0, 0]
#
#             siti_results_final[f'{name}_ti'] = siti_results_df['si']
#             siti_results_final[f'{name}_si'] = siti_results_df['ti']
#
#             """Join stats"""
#             siti_stats_json_final[name] = load_json(self.video_context.siti_stats)
#         # siti_results_final.to_csv(f'{self.video_context.siti_folder /
#         # "siti_results_final.csv"}', index_label='frame')
#         # pd.DataFrame(siti_stats_json_final).to_csv(f'{self.video_context.siti_folder
#         # / "siti_stats_final.csv"}')
#
#     def _scatter_plot_siti(self):
#         siti_results_df = pd.read_csv(
#             f'{self.video_context.siti_folder / "siti_stats_final.csv"}', index_col=0)
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
#         fig: plt.Figure
#         ax: plt.Axes
#         for column in siti_results_df:
#             si = siti_results_df[column]['si_2q']
#             ti = siti_results_df[column]['ti_2q']
#             name = column.replace('_nas', '')
#             ax.scatter(si, ti, label=name)
#         ax.set_xlabel("Spatial Information", fontdict={'size': 12})
#         ax.set_ylabel('Temporal Information', fontdict={'size': 12})
#         ax.set_title('Si/Ti', fontdict={'size': 16})
#         ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
#                   fontsize='small')
#         fig.tight_layout()
#         fig.savefig(self.video_context.siti_folder / 'scatter.png')
#         fig.show()
# </editor-fold>


class MakeViewport:
    def __init__(self,
                 config: str,
                 role: str,
                 ds_name: str,
                 overwrite=False):
        function = {'NAS_ERP': self.nas_erp,
                    }
        self.config = Config(config)
        self.role = Role(name=role, deep=0, operation=function[role])
        self.overwrite = overwrite
        self.dataset_name = ds_name

        self.video_context = VideoContext(self.config, deep=0)

        self.database_json = Path('datasets') / ds_name / f'{ds_name}.json'

        self.workfolder = self.video_context.viewport_folder / role.lower()
        self.workfolder_data = self.workfolder / 'data'
        self.workfolder_data.mkdir(parents=True, exist_ok=True)

        function[role]()

    def nas_erp(self):
        # database[video][user] = (yaw, pitch, roll)  # in rad
        database = load_json(self.database_json)

        for self.video_context.video in self.video_context.videos_list:
            video = self.video_context.video.name.replace('_erp_nas', '_nas')
            if video == 'wild_elephants_nas': continue
            for user in database[video]:
                yaw_pitch_roll_frames = database[video][user]
                for self.video_context.tiling in self.video_context.tiling_list:
                    tiling = f'{self.video_context.tiling}'
                    if tiling == '1x1': continue
                    get_tiles_json = self.video_context.get_tiles_json
                    input_video = self.workfolder / f'videos/{video}_erp_200x100_30.mp4'
                    output_video = self.workfolder / f'{video}_{user}_{tiling}.mp4'
                    if output_video.exists(): continue

                    erp = ERP(tiling, '576x288', '90x90')
                    height, width = erp.shape

                    reader = skvideo.io.FFmpegReader(f'{input_video}')
                    writer = skvideo.io.FFmpegWriter(output_video,
                                                     inputdict={'-r': '30'},
                                                     outputdict={'-r': '30', '-pix_fmt': 'yuv420p'})

                    #                          [ y ,  p ,   r]
                    # yaw_pitch_roll_frames = [[  0,   0,   0],
                    #                          [ 90,   0,   0],
                    #                          [-90,   0,   0],
                    #                          [  0,  45,   0],
                    #                          [  0, -45,   0],
                    #                          [  0,   0,  45],
                    #                          [  0,   0, -45],
                    #                          ]

                    get_tiles_data = load_json(get_tiles_json)
                    get_tiles = get_tiles_data['erp'][tiling][user]['tiles']
                    seen_tiles_chunk = get_tiles_data['erp'][tiling][user]['chunks']

                    print('\nDrawing viewport')
                    count = 0
                    for frame_array, (yaw, pitch, roll) in zip(reader, yaw_pitch_roll_frames):

                        frame_img = Image.fromarray(frame_array).resize((width, height))
                        yaw, pitch, roll = np.deg2rad((yaw, pitch, roll))
                        erp.set_vp(yaw, pitch, roll)

                        # Draw all tiles border
                        erp.clear_image()
                        erp.draw_all_tiles_borders(lum=200)
                        cover = Image.new("RGB", (width, height), (255, 0, 0))
                        frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

                        # Draw VP tiles by chunk
                        erp.clear_image()
                        chunk_idx = str(1 + count // 30)
                        for tile in seen_tiles_chunk[chunk_idx]:
                            erp.draw_tile_border(idx=tile, lum=200)
                        cover = Image.new("RGB", (width, height), (0, 255, 0))
                        frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))

                        # Draw viewport borders
                        erp.clear_image()
                        erp.draw_vp_borders(lum=255)
                        cover = Image.new("RGB", (width, height), (0, 0, 0))
                        frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(erp.projection))
                        # frame_img.show()

                        # noinspection PyTypeChecker
                        writer.writeFrame(np.array(frame_img))
                        print(f'\r{video}-tiling{tiling}-user{user}-frame{count}', end='')
                        count += 1

                        # todo: remove this in the future
                        if count >= 300: break

                    print('')
                    writer.close()
                break



class GetViewportQuality:
    """
    Essa classe vai pegar os tiles em que o usuário está vendo e vai extrair o
    viewport.
    """


from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


class Siti:
    pass


def total_size(o, handlers=None, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    if handlers is None:
        handlers = {}
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(obj):
        if id(obj) in seen:  # do not double count the same object
            return 0
        seen.add(id(obj))
        s = getsizeof(obj, default_size)

        if verbose:
            print(s, type(obj), repr(obj), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(obj, typ):
                s += sum(map(sizeof, handler(obj)))
                break
        return s

    return sizeof(o)
