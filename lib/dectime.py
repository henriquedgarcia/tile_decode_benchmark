from __future__ import print_function

from reprlib import repr
import datetime
import json
import time
from builtins import PermissionError
from collections import Counter, defaultdict
from contextlib import contextmanager
from logging import warning, debug, fatal
from pathlib import Path
from subprocess import run, DEVNULL
from typing import Any, NamedTuple, Union, Dict, List, Optional

import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import skvideo.io
from PIL import Image
from cycler import cycler
from fitter import Fitter

from .assets import AutoDict, Role
from .transform import cart2hcs
from .util import (run_command, check_video_gop, iter_frame, load_sph_file,
                   save_json, load_json, lin_interpol, save_pickle,
                   load_pickle, splitx, idx2xy)
from .video_state import Config, VideoContext
from .viewport import ERP


class BaseTileDecodeBenchmark:
    config: Config = None
    video_context: VideoContext = None
    role: Role = None

    def run(self, **kwargs):
        self.print_resume()
        self.role.init()

        for n in self.video_context:
            action = self.role.operation(**kwargs)

            if action in (None, 'continue', 'skip'):
                continue
            elif action in ('break',):
                break
            elif action in ('exit',):
                return

        self.role.finish()
        print(f'The end of {self.role.name}')

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.role.name}\n'
              f'  project: {self.video_context.project_path}\n'
              f'  codec: {self.config["codec"]}\n'
              f'  fps: {self.config["fps"]}\n'
              f'  gop: {self.config["gop"]}\n'
              f'  qualities: {self.config["quality_list"]}\n'
              f'  patterns: {self.config["tiling_list"]}'
              )
        print('=' * 70)

    def count_decoding(self) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :return:
        """
        dectime_log = self.video_context.dectime_log
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

    def get_times(self) -> List[float]:
        times = []
        dectime_log = self.video_context.dectime_log

        for line in dectime_log.read_text(encoding='utf-8').splitlines():
            dectime = line.strip().split(' ')[1].split('=')[1][:-1]
            times.append(float(dectime))

        content = self.video_context.dectime_log.read_text(encoding='utf-8')
        content_lines = content.splitlines()
        times = [float(line.strip().split(' ')[1].split('=')[1][:-1])
                 for line in content_lines if 'utime' in line]
        return times


class TileDecodeBenchmark(BaseTileDecodeBenchmark):
    results = AutoDict()
    results_dataframe = pd.DataFrame()

    def __init__(self, config: str = None, role: str = None, stub=False, **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        if stub: return
        operations = {
            'PREPARE': Role(name='PREPARE', deep=1, init=None,
                            operation=self.prepare, finish=None),
            'COMPRESS': Role(name='COMPRESS', deep=4, init=None,
                             operation=self.compress, finish=None),
            'SEGMENT': Role(name='SEGMENT', deep=4, init=None,
                            operation=self.segment, finish=None),
            'DECODE': Role(name='DECODE', deep=5, init=None,
                           operation=self.decode, finish=None),
            'COLLECT_RESULTS': Role(name='COLLECT_RESULTS', deep=5, init=None,
                                    operation=self.collect_dectime,
                                    finish=self.save_dectime),
        }

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    # PREPARE
    def prepare(self, overwrite=False) -> Any:
        """
        Prepare video to encode. Uncompress and unify resolution, frame rate,
        pixel format.
        :param overwrite: If True, this method overwrite previous files.
        """
        original = self.video_context.original_file
        uncompressed_file = self.video_context.lossless_file
        lossless_log = self.video_context.lossless_file.with_suffix('.log')

        debug(f'==== Processing {uncompressed_file} ====')

        if uncompressed_file.exists() and not overwrite:
            warning(f'The file {uncompressed_file} exist. Skipping.')
            return 'skip'

        if not original.exists():
            warning(f'The file {original} not exist. Skipping.')
            return 'skip'

        video = self.video_context.video
        resolution = self.video_context.video.resolution
        dar = resolution.W / resolution.H

        cmd = f'ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {video.offset} '
        cmd += f'-i {original} '
        cmd += f'-crf 0 '
        cmd += f'-t {video.duration} '
        cmd += f'-r {video.fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf "scale={video.resolution},setdar={dar}" '
        cmd += f'{uncompressed_file}'

        run_command(cmd, lossless_log, 'w')

    # COMPRESS
    def compress(self, overwrite=False) -> Any:
        """
        Encode videos using h.265
        :param overwrite:
        :return:
        """
        uncompressed_file = self.video_context.lossless_file
        compressed_file = self.video_context.compressed_file
        compressed_log = self.video_context.compressed_file.with_suffix('.log')

        debug(f'==== Processing {compressed_file} ====')

        if compressed_file.exists() and not overwrite:
            warning(f'The file {compressed_file} exist. Skipping.')
            return 'skip'

        if not uncompressed_file.exists():
            warning(f'The file {uncompressed_file} not exist. Skipping.')
            return 'skip'

        quality = self.video_context.quality
        gop = self.video_context.video.gop
        tile = self.video_context.tile

        cmd = ['ffmpeg -hide_banner -y -psnr']
        cmd += [f'-i {uncompressed_file}']
        cmd += [f'-crf {quality} -tune "psnr"']
        cmd += [f'-c:v libx265']
        cmd += [f'-x265-params']
        cmd += [f'"keyint={gop}:'
                f'min-keyint={gop}:'
                f'open-gop=0:'
                f'scenecut=0:'
                f'info=0"']
        cmd += [f'-vf "crop='
                f'w={tile.resolution.W}:h={tile.resolution.H}:'
                f'x={tile.position.x}:y={tile.position.y}"']
        cmd += [f'{compressed_file}']
        cmd = ' '.join(cmd)

        run_command(cmd, compressed_log, 'w')

    # SEGMENT
    def segment(self, overwrite=False) -> Any:
        segment_log = self.video_context.segment_file.with_suffix('.log')
        segment_folder = self.video_context.segment_folder
        compressed_file = self.video_context.compressed_file

        # info(f'==== Processing {segment_folder} ====')

        if segment_log.is_file() and segment_log.stat().st_size > 10000 \
                and not overwrite:
            # If segment log size is very small, infers error and overwrite.
            warning(f'The file {segment_log} exist. Skipping.')
            return 'skip'

        if not compressed_file.is_file():
            warning(f'The file {compressed_file} not exist. Skipping.')
            return 'skip'

        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4

        cmd = ['MP4Box']
        cmd += ['-split 1']
        cmd += [f'{compressed_file}']
        cmd += [f'-out {segment_folder}{Path("/")}']
        cmd = ' '.join(cmd)
        cmd = f'bash -c "{cmd}"'
        return run_command, (cmd, segment_log, 'w')

    # DECODE
    def decode(self, overwrite=False) -> Any:
        segment_file = self.video_context.segment_file
        dectime_log = self.video_context.dectime_log
        # info(f'==== Processing {dectime_log} ====')

        diff = self.video_context.decoding_num
        if self.video_context.dectime_log.exists():
            count = self.count_decoding()
            diff = self.video_context.decoding_num - count
            if diff <= 0 and not overwrite:
                warning(f'{segment_file} is decoded enough. Skipping.')
                return 'skip'

        if not segment_file.is_file():
            warning(f'The file {segment_file} not exist. Skipping.')
            return 'skip'

        cmd = (f'ffmpeg -hide_banner -benchmark '
               f'-codec hevc -threads 1 '
               f'-i {segment_file} '
               f'-f null -')
        for i in range(diff):
            run_command(cmd, dectime_log, 'a')

    # COLLECT RESULTS
    def collect_dectime(self, overwrite=False) -> Any:
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
        :param overwrite:
        :return:
        """
        debug(f'Collecting {self.video_context}')
        dectime_json_file = self.video_context.dectime_json_file
        segment_file = self.video_context.segment_file
        dectime_log = self.video_context.dectime_log

        if dectime_json_file.exists() and not overwrite:
            warning(f'The file {dectime_json_file} exist and not overwrite. Skipping.')
            return 'exit'

        if not segment_file.exists():
            warning(f'The chunk {self.video_context.segment_file} not exist. Skipping.')
            return 'skip'

        if not dectime_log.exists():
            warning(f'The dectime log {self.video_context.dectime_log} not exist. Skipping.')
            return 'skip'

        try:
            chunk_size = segment_file.stat().st_size
            bitrate = 8 * chunk_size / self.video_context.video.chunk_dur
        except PermissionError:
            warning(f'PermissionError error on reading size of '
                    f'{self.video_context.segment_file}. Skipping.')
            return 'skip'

        dectimes: List[float] = self.get_times()

        data = {'dectimes': dectimes, 'bitrate': bitrate}
        print(f'\r{data}', end='')

        results = self.results
        for factor in self.video_context.state:
            results = results[factor]

        results.update(data)

    def save_dectime(self):
        filename = self.video_context.dectime_json_file
        # info(f'Saving {filename}')
        save_json(self.results, filename)
        filename = self.video_context.dectime_json_file.with_suffix('.pickle')
        save_pickle(self.results, filename)


class Graphs:
    config: Config = None
    video_context: VideoContext = None
    _video: str = None
    quality_ref: str = '0'
    quality: str = None
    tiling: str = None
    metric: str = None
    tile: int = None
    chunk: int = None
    proj: str = None
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

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value: str):
        self._video = value

    @property
    def name(self):
        name = self.video.replace('_cmp', '').replace('_erp', '')
        return name

    @property
    def name_list(self):
        return list(set([video.replace('_cmp', '').replace('_erp', '') for video in self.videos_list]))

    @property
    def proj_list(self):
        projs = set()
        for video in self.videos_list:
            projection: str = self.videos_list[video]['projection']
            projs.update((projection,))
        return list(projs)

    @property
    def vid_proj(self):
        projection: str = self.videos_list[self.video]['projection']
        return projection

    @property
    def videos_list(self):
        return self.config.videos_list

    @property
    def tiling_list(self):
        return self.config['tiling_list']

    @property
    def metric_list(self):
        return ['time', 'rate', 'time_std', 'PSNR', 'WS-PSNR', 'S-PSNR']

    @property
    def quality_list(self):
        quality_list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list

    @property
    def tile_list(self):
        tile_m, tile_n = list(map(int, self.tiling.split('x')))
        total_tiles = tile_n * tile_m
        return list(range(total_tiles))

    @property
    def chunk_list(self):
        duration = self.config.videos_list[self.video]['duration']
        return list(range(1, int(duration) + 1))

    @property
    def duration(self):
        return self.config.videos_list[self.video]['duration']

    @contextmanager
    def dectime_ctx(self):
        dectime = {}
        try:
            dectime_filename = self.video_context.project_path / self.video_context.dectime_folder / f'dectime_{self.video}.json'
            dectime = load_json(dectime_filename, object_hook=dict)
            yield dectime
        finally:
            del dectime

    @contextmanager
    def quality_ctx(self):
        quality = {
            'tiling': {
                'crf': {
                    'tile': {
                        'psnr': [],
                        'ws-psnr': [],
                        's-psnr': [],
                    },
                },
            },
        }
        try:
            dectime_filename = self.video_context.project_path / self.video_context.quality_folder / f'quality_{self.video}.json'
            quality = load_json(dectime_filename, object_hook=dict)
            yield quality
        finally:
            del quality


class DectimeGraphs(BaseTileDecodeBenchmark, Graphs):
    # Fitter
    fit: Fitter
    fitter = AutoDict()
    fit_errors = AutoDict()
    bins: Union[str, int]
    data: Union[dict, AutoDict] = None

    def __init__(self,
                 config: str,
                 role: str,
                 bins: List[Union[str, int]],
                 **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        function = {'BY_PATTERN': self.by_pattern,
                    'BY_PATTERN_BY_QUALITY': self.by_pattern_by_quality,
                    'BY_VIDEO_BY_PATTERN_BY_QUALITY': self.by_video_by_tiling_by_quality,
                    'BY_PATTERN_FULL_FRAME': self.by_pattern_full_frame,
                    }

        self.config = Config(config)
        self.role = Role(name=role, deep=0, operation=function[role])
        self.video_context = VideoContext(self.config, self.role.deep)
        self.rc_config()

        self.workfolder = self.video_context.graphs_folder / role
        self.workfolder_data = self.workfolder / 'data'
        self.workfolder_data.mkdir(parents=True, exist_ok=True)

        for self.bins in bins:
            self.print_resume()
            function[role](**kwargs)
            print(f'\n====== The end of {self.role.name} ======')

    @staticmethod
    def rc_config():
        import matplotlib as mpl
        rc_param = {"figure": {'figsize': (12.8, 3.84), 'dpi': 300, 'autolayout': False},
                    "axes": {'linewidth': 0.5, 'titlesize': 8, 'labelsize': 6,
                             'prop_cycle': cycler(color=[plt.get_cmap('tab20')(i) for i in range(20)])},
                    "xtick": {'labelsize': 6},
                    "ytick": {'labelsize': 6},
                    "legend": {'fontsize': 6},
                    "font": {'size': 8},
                    "patch": {'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': '#3297c9'},
                    "lines": {'linewidth': 0.5, 'markersize': 3},
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
    def make_graph(plot_type: Union[str, str], ax: axes.Axes,
                   x: List[float] = None, y: List[float] = None,
                   yerr: List[float] = None,
                   legend: dict = None, title: str = None, label: str = None,
                   xlabel: str = None, ylabel: str = None,
                   xticklabels: list[Union[float, str]] = None, xticks: list[float] = None,
                   width: int = 0.8,
                   scilimits: Union[tuple, Optional[tuple[tuple[str, tuple[int, int]], ...]]] = None,
                   color=None, bins=None, histtype='bar', cumulative=False,
                   log=False):

        if plot_type == 'hist':
            artist = ax.hist(y, bins=bins, histtype=histtype, label=label,
                             cumulative=cumulative, density=True, log=log,
                             color='#bbbbbb', edgecolor='black', linewidth=0.2)

        elif plot_type == 'bar':
            artist = ax.bar(x, y, width=width, yerr=yerr, label=label, color=color)

        elif plot_type == 'plot':
            color = color if color else None
            artist = ax.plot(x, y, color=color, linewidth=1, label=label)

        elif plot_type == 'boxplot':
            color = color if color else None
            artist = ax.boxplot(y, positions=x, widths=width,
                                whis=(0, 100),
                                showfliers=False,
                                boxprops=dict(facecolor=color),
                                flierprops=dict(color='r'),
                                medianprops=dict(color='k'),
                                patch_artist=True, labels=label)
            for cap in artist['caps']:
                cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

            # # Get numbers of ouliers
            # for xtic, line in zip(xticks, bx['fliers']):
            #     print(f'{title} - CRF {xtic} {len(line.get_xdata())}')
        else:
            raise ValueError(f'The plot_type parameter "{plot_type}" not supported.')

        if scilimits is not None:
            for axis, limits in scilimits:
                ax.ticklabel_format(axis=axis,
                                    style='scientific',
                                    scilimits=limits)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        if title is not None:
            ax.set_title(title)
        if legend is not None:
            ax.legend(**legend)

        return artist

    def by_pattern(self, overwrite, n_dist):
        class ProjectPaths:
            ctx = self

            @property
            def quality_file(self) -> Path:
                quality_file = self.ctx.video_context.project_path / self.ctx.video_context.quality_folder / f'quality_{self.ctx.video}.json'
                return quality_file

            @property
            def dectime_file(self) -> Path:
                dectime_file = self.ctx.video_context.project_path / self.ctx.video_context.dectime_folder / f'dectime_{self.ctx.video}.json'
                return dectime_file

            @property
            def data_file(self) -> Path:
                data_file = self.ctx.workfolder_data / f'data.json'
                return data_file

            @property
            def fitter_pickle_file(self) -> Path:
                fitter_file = self.ctx.workfolder_data / f'fitter_{self.ctx.proj}_{self.ctx.tiling}_{self.ctx.metric}_{self.ctx.bins}bins.pickle'
                return fitter_file

            @property
            def stats_file(self) -> Path:
                stats_file = self.ctx.workfolder / f'stats_{self.ctx.bins}bins.csv'
                return stats_file

            @property
            def corretations_file(self) -> Path:
                corretations_file = self.ctx.workfolder / f'correlations.csv'
                return corretations_file

            @property
            def hist_pattern_file(self) -> Path:
                img_file = self.ctx.workfolder / f'hist_pattern_{self.ctx.proj}_{self.ctx.metric}_{self.ctx.bins}bins.png'
                return img_file

            @property
            def bar_pattern_file(self) -> Path:
                img_file = self.ctx.workfolder / f'bar_pattern_{self.ctx.metric}.png'
                return img_file

            @property
            def boxplot_pattern_file(self) -> Path:
                img_file = self.ctx.workfolder / f'boxplot_pattern_{self.ctx.metric}.png'
                return img_file

        paths = ProjectPaths()

        # overwrite = True

        def main(overwrite=True):
            get_data(overwrite, remove_outliers=True)
            # make_fit()
            # calc_stats()
            # make_hist(overwrite)
            make_bar(overwrite)
            # make_boxplot(overwrite)
            # make_violinplot(overwrite)

        def get_data(overwrite=False, remove_outliers=False):
            print('\n====== Get Data ======')
            # data[self.proj][self.tiling][self.metric]
            data_file = paths.data_file

            if data_file.exists() and not overwrite:
                warning(f'\n  The data file "{data_file}" exist. Loading date.')
                self.data = load_json(paths.data_file, object_hook=dict)
                return

            self.data = AutoDict()

            for self.video in self.videos_list:
                dectime = load_json(paths.dectime_file, object_hook=dict)
                qlt_data = load_json(paths.quality_file, object_hook=dict)

                for self.tiling in self.tiling_list:
                    print(f'\r  Getting - {self.vid_proj}  {self.name} {self.tiling} ... ', end='')

                    bucket = self.data[self.vid_proj][self.tiling]
                    if not isinstance(bucket['time'], list):
                        bucket = self.data[self.vid_proj][self.tiling] = defaultdict(list)

                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                # quality_val["PSNR"|"WS-PSNR"|"S-PSNR"]: float
                                # dectime_val["dectimes"|"bitrate"]: list[float]|int

                                dectime_val = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                quality_val = qlt_data[self.tiling][self.quality][f'{self.tile}']

                                bucket['time'].append(float(np.average(dectime_val['dectimes'])))
                                bucket['rate'].append(float(dectime_val['bitrate']))
                                bucket['time_std'].append(float(np.std(dectime_val['dectimes'])))

                                for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                    value = quality_val[metric]
                                    idx = self.chunk - 1
                                    avg_qlt = float(np.average(value[idx * 30: idx * 30 + 30]))
                                    if avg_qlt == float('inf'):
                                        avg_qlt = 1000
                                    bucket[metric].append(avg_qlt)

                    print('OK')

            #### Fliers analisys data[self.proj][self.tiling][self.metric]
            def remove_outliers(data):
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
                                                 if (clean_left <= d
                                                     and d <= clean_right)]
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

                pd.DataFrame(resume).to_csv(data_file.with_name('resume_data_clean.csv'))

            if remove_outliers:
                remove_outliers(self.data)

            print(f' Saving... ', end='')
            save_json(self.data, data_file)
            print(f'  Finished.')

        def make_fit(overwrite=False):
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')

            # Load data file
            if self.data is None:
                self.data = load_json(paths.data_file, object_hook=dict)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    for self.metric in self.metric_list:
                        print(f'  Fitting - {self.proj} {self.tiling} {self.metric}... ', end='')

                        # Check fitter pickle
                        if paths.fitter_pickle_file.exists() and not overwrite:
                            print(f'Pickle found! Skipping.')
                            continue

                        # Load data file
                        samples = self.data[self.proj][self.tiling][self.metric]

                        # Make the fit
                        distributions = self.config['distributions']
                        fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                        fitter.fit()

                        # Saving
                        print(f'  Saving... ')
                        save_pickle(fitter, paths.fitter_pickle_file)
                        print(f'  Finished.')

        def calc_stats(overwrite=False):
            print('  Calculating Statistics')

            # Check stats file
            if paths.stats_file.exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                return

            # Load data file
            if self.data is None:
                self.data = load_json(paths.data_file, object_hook=dict)
            stats = defaultdict(list)
            corretations_bucket = defaultdict(list)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    # Load data
                    data_bucket = self.data[self.proj][self.tiling]

                    for self.metric in self.metric_list:
                        # Load fitter pickle
                        fitter = load_pickle(paths.fitter_pickle_file)

                        # Calculate percentiles
                        percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
                        df_errors: pd.DataFrame = fitter.df_errors

                        # Calculate errors
                        sse: pd.Series = df_errors['sumsquare_error']
                        bins = len(fitter.x)
                        rmse = np.sqrt(sse / bins)
                        nrmse = rmse / (sse.max() - sse.min())

                        # Append info and stats on Dataframe
                        stats[f'proj'].append(self.proj)
                        stats[f'tiling'].append(self.tiling)
                        stats[f'metric'].append(self.metric)
                        stats[f'bins'].append(bins)
                        stats[f'average'].append(np.average(data_bucket[self.metric]))
                        stats[f'std'].append(float(np.std(data_bucket[self.metric])))
                        stats[f'min'].append(percentile[0])
                        stats[f'quartile1'].append(percentile[1])
                        stats[f'median'].append(percentile[2])
                        stats[f'quartile3'].append(percentile[3])
                        stats[f'max'].append(percentile[4])

                        # Append distributions on Dataframe
                        for dist in sse.keys():
                            params = fitter.fitted_param[dist]
                            dist_info = self.find_dist(dist, params)

                            stats[f'rmse_{dist}'].append(rmse[dist])
                            stats[f'nrmse_{dist}'].append(nrmse[dist])
                            stats[f'sse_{dist}'].append(sse[dist])
                            stats[f'param_{dist}'].append(dist_info['parameters'])
                            stats[f'loc_{dist}'].append(dist_info['loc'])
                            stats[f'scale_{dist}'].append(dist_info['scale'])

                    for self.metric in self.metric_list:
                        if self.metric != 'time':
                            corr_time_metric = np.corrcoef((data_bucket['time'], data_bucket[self.metric]))[1][0]
                        else:
                            corr_time_metric = 1.0
                        if self.metric != 'rate':
                            corr_rate_metric = np.corrcoef((data_bucket['rate'], data_bucket[self.metric]))[1][0]
                        else:
                            corr_rate_metric = 1.0
                        if self.metric != 'PSNR':
                            corr_psnr_metric = np.corrcoef((data_bucket['PSNR'], data_bucket[self.metric]))[1][0]
                        else:
                            corr_psnr_metric = 1.0
                        if self.metric != 'WS-PSNR':
                            corr_wspsnr_metric = np.corrcoef((data_bucket['WS-PSNR'], data_bucket[self.metric]))[1][0]
                        else:
                            corr_wspsnr_metric = 1.0
                        if self.metric != 'S-PSNR':
                            corr_spsnr_metric = np.corrcoef((data_bucket['S-PSNR'], data_bucket[self.metric]))[1][0]
                        else:
                            corr_spsnr_metric = 1.0

                        corretations_bucket[f'proj'].append(self.proj)
                        corretations_bucket[f'tiling'].append(self.tiling)
                        corretations_bucket[f'metric'].append(self.metric)
                        corretations_bucket[f'corr_time_metric'].append(corr_time_metric)
                        corretations_bucket[f'corr_rate_metric'].append(corr_rate_metric)
                        corretations_bucket[f'corr_psnr_metric'].append(corr_psnr_metric)
                        corretations_bucket[f'corr_wspsnr_metric'].append(corr_wspsnr_metric)
                        corretations_bucket[f'corr_spsnr_metric'].append(corr_spsnr_metric)

            pd.DataFrame(stats).to_csv(str(paths.stats_file), index=False)
            pd.DataFrame(corretations_bucket).to_csv(str(paths.corretations_file), index=False)

        def make_hist(overwrite=False):
            print(f'\n====== Make Plot - Bins = {self.bins} ======')

            # make an image for each metric
            for self.metric in self.metric_list:
                for self.proj in self.proj_list:
                    # Check image file by metric
                    im_file = paths.hist_pattern_file
                    if im_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # Make figure
                    fig_pdf = figure.Figure(figsize=(12.8, 1.92), dpi=300)  # pdf
                    fig_cdf = figure.Figure(figsize=(12.8, 1.92), dpi=300)  # cdf
                    pos = [(1, 5, x) for x in range(1, 5 * 1 + 1)]
                    subplot_pos = iter(pos)

                    for self.tiling in self.tiling_list:
                        # Load fitter and select samples
                        fitter = load_pickle(paths.fitter_pickle_file)

                        # Position of plot
                        nrows, ncols, index = next(subplot_pos)
                        ax_pdf: axes.Axes = fig_pdf.add_subplot(nrows, ncols, index)
                        ax_cdf: axes.Axes = fig_cdf.add_subplot(nrows, ncols, index)

                        # Make histogram
                        kwrds = dict(label='empirical', color='#dbdbdb', width=fitter.x[1] - fitter.x[0])
                        bins_area = [y * (fitter.x[1] - fitter.x[0]) for y in fitter.y]
                        ax_pdf.bar(fitter.x, fitter.y, **kwrds)
                        ax_cdf.bar(fitter.x, np.cumsum(bins_area), **kwrds)

                        # make plot for n_dist distributions
                        dists = fitter.df_errors['sumsquare_error'].sort_values()[0:n_dist].index
                        for dist_name in dists:
                            fitted_pdf = fitter.fitted_pdf[dist_name]

                            dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                            param = fitter.fitted_param[dist_name]
                            cdf_fitted = dist.cdf(fitter.x, *param)

                            sse = fitter.df_errors['sumsquare_error'][dist_name]
                            label = f'{dist_name} - SSE {sse: 0.3e}'

                            ax_pdf.plot(fitter.x, fitted_pdf, color=self.color_list[dist_name], linewidth=1, label=label)
                            ax_cdf.plot(fitter.x, cdf_fitted, color=self.color_list[dist_name], linewidth=1, label=label)

                        scilimits = (-3, -3) if self.metric == 'time' else (6, 6)
                        title = f'{self.proj.upper()}-{self.tiling}'
                        xlabel = 'Decoding time (ms)' if self.metric == 'time' else 'Bit Rate (Mbps)'
                        ylabel = 'Density' if index in [1, 6] else None
                        ylabel2 = 'Cumulative' if index in [1, 6] else None
                        legkwrd_pdf = {'loc': 'upper right'}
                        legkwrd_cdf = {'loc': 'lower right'}

                        ax_pdf.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax_pdf.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                        ax_cdf.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                        ax_cdf.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                        ax_pdf.set_title(title)
                        ax_cdf.set_title(title)
                        ax_pdf.set_xlabel(xlabel)
                        ax_cdf.set_xlabel(xlabel)
                        # ax.set_ylabel(ylabel)
                        # ax2.set_ylabel(ylabel2)
                        # ax.set_yscale('log')
                        ax_pdf.legend(*legkwrd_pdf)
                        ax_cdf.legend(*legkwrd_cdf)

                    print(f'  Saving the figure')
                    fig_pdf.savefig(im_file)
                    fig_cdf.savefig(im_file.with_stem(f'{im_file.stem}_cdf'))

        def make_bar(overwrite=False):
            print(f'\n====== Make Bar - Bins = {self.bins} ======')

            figsize = (6.4, 1.92)
            pos = [(1, 5, x) for x in range(1, 1 * 5 + 1)]
            scilimits_y = {'time': (-3, -3), 'time_std': (-3, -3), 'rate': (6, 6)}
            label_y = {'time': 'Average decoding time (ms)',
                       'time_std': 'Std dev of decoding time (ms)',
                       'rate': 'Bit Rate (Mbps)',
                       'PSNR': 'PSNR',
                       'WS-PSNR': 'WS-PSNR',
                       'S-PSNR': 'S-PSNR'}
            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            legend_handles = [mpatches.Patch(color=colors['cmp'], label='CMP'),
                              mpatches.Patch(color=colors['erp'], label='ERP'), ]

            stats = pd.read_csv(paths.stats_file)

            for self.metric in self.metric_list:
                path = paths.bar_pattern_file
                if path.exists() and not overwrite:
                    warning(f'Figure exist. Skipping')
                    return

                fig = figure.Figure(figsize=figsize)
                subplot_pos = iter(pos)

                # for start, self.proj in enumerate(self.proj_list):
                for (nrows, ncols, index), self.tiling in zip(subplot_pos, self.tiling_list):
                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                    # Ploting the bars of tiling XXX.
                    for x, self.proj in enumerate(['cmp', 'erp']):
                        data = stats[(stats[f'metric'] == self.metric)]
                        data = data[(data[f'tiling'] == self.tiling)]
                        data = data[(data[f'proj'] == self.proj)]
                        height = data[f'average']
                        yerr = data[f'std']
                        ax.bar(x, height, width=0.8, yerr=yerr, color=colors[self.proj])

                    # Finishing of Graphs
                    if index == 1:
                        ax.set_ylabel(label_y[self.metric])
                    ax.set_xticks([0.5])
                    ax.set_xticklabels([self.tiling])
                    if self.metric in ['time', 'rate', 'time_std']:
                        ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                    if index == 3:
                        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', ncol=2)

                fig.suptitle(f'Average and Standard Deviation of {self.metric}')
                fig.tight_layout(pad=0.8, h_pad=0, w_pad=0, rect=(0.0, 0.0, 1.0, 1.0))

                print(f'Saving the figure')
                fig.savefig(path)

        def make_boxplot(overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')
            # overwrite = True
            # Check image file by metric
            for self.metric in self.metric_list:
                img_file = paths.boxplot_pattern_file
                if img_file.exists() and not overwrite:
                    warning(f'Figure exist. Skipping')
                    return

                colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
                label_y = {'time': 'Decoding time (ms)', 'rate': 'Bit Rate (Mbps)'}
                scilimits_y = {'time': (-3, -3), 'rate': (6, 6)}
                xticks = [i + 0.5 for i in range(0, len(self.tiling_list) * 3, 3)]
                xticklabels = self.tiling_list
                n_ticks = len(self.tiling_list)
                bar_by_ticks = len(self.proj_list) + 1
                legend_handles = [mpatches.Patch(color=colors['cmp'], label='CMP'),
                                  mpatches.Patch(color=colors['erp'], label='ERP'), ]

                if self.data is None:
                    self.data = load_json(paths.data_file, object_hook=dict)

                positions = lambda bar_id: [x for x in range(bar_id, n_ticks * bar_by_ticks, bar_by_ticks)]

                fig = figure.Figure(figsize=(5.8, 1.92))
                pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
                subplot_pos = iter(pos)

                fig_boxplot_separated = figure.Figure(figsize=(6.8, 3.84))
                pos_sep = [(2, 5, x) for x in range(1, 2 * 5 + 1)]
                subplot_pos_sep = iter(pos_sep)

                nrows, ncols, index = next(subplot_pos)
                ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                for bar_offset, self.proj in enumerate(self.proj_list):
                    ################ get heights and positions ################
                    data_by_tiling = [self.data[self.proj][self.tiling][self.metric] for self.tiling in self.tiling_list]

                    ################ joined axes ################
                    boxplot = ax.boxplot(data_by_tiling, positions=positions(bar_offset), widths=0.8,
                                         whis=(0, 100), showfliers=False,
                                         boxprops=dict(facecolor=colors[self.proj]),
                                         flierprops=dict(color='r'),
                                         medianprops=dict(color='k'),
                                         patch_artist=True)
                    for cap in boxplot['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                # Finish normal labels
                ax.set_ylabel(label_y[self.metric])
                ax.set_title(f'{self.metric.capitalize()}')
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                ax.legend(handles=legend_handles, loc='upper right')

                ################ separated axes ################
                for bar_offset, self.tiling in enumerate(self.tiling_list):
                    nrows_sep, ncols_sep, index_sep = next(subplot_pos_sep)
                    ax_sep: axes.Axes = fig_boxplot_separated.add_subplot(nrows_sep, ncols_sep, index_sep)
                    box_positions = {'erp': 2, 'cmp': 1}

                    for self.proj in self.proj_list:
                        tiling_data = self.data[self.proj][self.tiling][self.metric]
                        boxplot_sep = ax_sep.boxplot((tiling_data,), positions=(box_positions[self.proj],), widths=0.8,
                                                     whis=(0, 100), showfliers=False,
                                                     boxprops=dict(facecolor=colors[self.proj]),
                                                     flierprops=dict(color='r'),
                                                     medianprops=dict(color='k'),
                                                     patch_artist=True)
                        for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                    ax_sep.set_title(f'{self.metric.capitalize()}')
                    ax_sep.set_ylabel(label_y[self.metric])
                    ax_sep.set_xticks([1.5])
                    ax_sep.set_xticklabels([xticklabels[bar_offset]])
                    ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                    if index_sep in [5, 10]:
                        ax_sep.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')

                print(f'  Saving the figure')
                fig.savefig(img_file)
                stem = img_file.stem
                fig_boxplot_separated.savefig(img_file.with_stem(stem + '_sep'))

        def make_violinplot(overwrite=False):
            print(f'\n====== Make BoxPlot - Bins = {self.bins} ======')

            img_file = self.workfolder / f'violinplot_pattern.png'

            if img_file.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            colors = {'cmp': 'tab:green', 'erp': 'tab:blue'}
            label_y = {'time': 'Decoding time (ms)', 'rate': 'Bit Rate (Mbps)'}
            scilimits_y = {'time': (-3, -3), 'rate': (6, 6)}
            xticklabels = self.tiling_list
            legend_handles = [mpatches.Patch(color=colors['cmp'], label='CMP'),
                              mpatches.Patch(color=colors['erp'], label='ERP'), ]

            if self.data is None:
                self.data = load_json(paths.data_file, object_hook=dict)

            fig_boxplot_separated = figure.Figure(figsize=(6.8, 3.84))
            pos_sep = [(2, 5, x) for x in range(1, 2 * 5 + 1)]
            subplot_pos_sep = iter(pos_sep)

            for self.metric in self.metric_list:
                for bar_offset, self.tiling in enumerate(self.tiling_list):
                    nrows_sep, ncols_sep, index_sep = next(subplot_pos_sep)
                    ax_sep: axes.Axes = fig_boxplot_separated.add_subplot(nrows_sep, ncols_sep, index_sep)

                    tiling_data = [self.data[self.proj][self.tiling][self.metric] for self.proj in self.proj_list]
                    ax_sep.violinplot(tiling_data,
                                      positions=[0, 1],
                                      showmedians=True,
                                      widths=0.8)

                    ax_sep.set_title(f'{self.metric.capitalize()}')
                    ax_sep.set_ylabel(label_y[self.metric])
                    ax_sep.set_xticks([1.5])
                    ax_sep.set_xticklabels([xticklabels[bar_offset]])
                    ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                    if index_sep in [5, 10]:
                        ax_sep.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')

            print(f'  Saving the figure')
            fig_boxplot_separated.savefig(img_file)

        main()

    def by_pattern_by_quality(self, overwrite, n_dist):
        dataset_name = 'nasrabadi_28videos'
        class ProjectPaths:
            ctx = self

            @property
            def quality_file(self) -> Path:
                quality_file = self.ctx.video_context.project_path / self.ctx.video_context.quality_folder / f'quality_{self.ctx.video}.json'
                return quality_file

            @property
            def dectime_file(self) -> Path:
                dectime_file = self.ctx.video_context.project_path / self.ctx.video_context.dectime_folder / f'dectime_{self.ctx.video}.json'
                return dectime_file

            @property
            def get_tiles_json_file(self) -> Path:
                get_tiles_path = self.ctx.video_context.project_path / self.ctx.video_context.get_tiles_folder
                result_json_name = f'get_tiles_{dataset_name}_{self.ctx.video}_{self.ctx.tiling}.json'
                get_tiles_json = get_tiles_path / result_json_name
                return get_tiles_json

            @property
            def data_file(self) -> Path:
                data_file = self.ctx.workfolder_data / f'data.json'
                return data_file

            @property
            def fitter_pickle_file(self) -> Path:
                fitter_file = self.ctx.workfolder_data / f'fitter_{self.ctx.proj}_{self.ctx.tiling}_{self.ctx.quality}_{self.ctx.metric}_{self.ctx.bins}bins.pickle'
                return fitter_file

            @property
            def stats_file(self) -> Path:
                stats_file = self.ctx.workfolder / f'stats_{self.ctx.bins}bins.csv'
                return stats_file

            @property
            def corretations_file(self) -> Path:
                corretations_file = self.ctx.workfolder / f'correlations.csv'
                return corretations_file

            @property
            def boxplot_pattern_quality_file(self) -> Path:
                img_file = self.ctx.workfolder / f'boxplot_pattern_tiling_{self.ctx.tiling}_{self.ctx.metric}.png'
                return img_file

        paths = ProjectPaths()

        def main(overwrite=True):
            get_data()
            make_fit()
            calc_stats()
            make_hist()
            make_bar_tiling_quality()
            make_bar_quality_tiling()
            make_boxplot(overwrite)

        def get_data(overwrite=False):
            print('\n\n====== Get Data ======')
            # data[self.proj][self.tiling][self.quality][self.metric]
            data_file = paths.data_file

            if data_file.exists() and not overwrite:
                warning(f'\n  The data file "{data_file}" exist. Loading date.')
                return

            data = AutoDict()

            for self.video in self.videos_list:
                dectime = load_json(paths.dectime_file, object_hook=dict)
                qlt_data = load_json(paths.quality_file, object_hook=dict)

                for self.tiling in self.tiling_list:
                    get_tiles = load_json(paths.get_tiles_json_file)
                    get_tiles = get_tiles[self.vid_proj][self.tiling]
                    # get_tiles[proj][tiling][user]['tiles']
                    bucket = data[self.vid_proj][self.tiling]

                    for user in get_tiles:
                        chunk = 0
                        tiles_for_chunks = set()

                        if isinstance(bucket['get_tiles'][user][f'{chunk}'], list):
                            continue

                        for frame, vptiles in enumerate(get_tiles[user]['tiles']):
                            tiles_for_chunks.update(vptiles)
                            if (frame + 1) % 30 == 0:
                                bucket['get_tiles'][user][f'{chunk}'] = list(tiles_for_chunks)
                                tiles_for_chunks = set()
                                chunk += 1

                    for self.quality in self.quality_list:
                        print(f'\r  Getting - {self.vid_proj} {self.name} {self.tiling} {self.quality} ... ', end='')

                        if not isinstance(bucket['dectimes'][self.quality]['time'], list):
                            bucket['dectimes'][self.quality] = defaultdict(list)
                            bucket['seen_tiles'][self.quality] = defaultdict(list)
                            bucket['frame_tiles'][self.quality] = defaultdict(list)

                        for self.tile in self.tile_list:
                            # quality_val["psnr"|"ws-psnr"|"s-psnr"]: float
                            quality_val = qlt_data[self.tiling][self.quality][f'{self.tile}']

                            # Aqui eu coloco todos os tiles e todos os chunks
                            for self.chunk in self.chunk_list:
                                # dectime_val["dectimes"|"bitrate"]: list[float]|int
                                dectime_val = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                bucket['dectimes'][self.quality]['time'].append(float(np.average(dectime_val['dectimes'])))
                                bucket['dectimes'][self.quality]['rate'].append(float(dectime_val['bitrate']))
                                bucket['dectimes'][self.quality]['time_std'].append(float(np.std(dectime_val['dectimes'])))

                                for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                    value = quality_val[metric]
                                    idx = self.chunk - 1
                                    avg_qlt = float(np.average(value[idx * 30: idx * 30 + 30]))
                                    if avg_qlt == float('inf'):
                                        avg_qlt = 1000
                                    bucket['dectimes'][self.quality][metric].append(avg_qlt)

                            # aqui eu só coloco os chunks e tiles vistos por cada usuário
                            for self.chunk in self.chunk_list:
                                dectime_val = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                time = []
                                rate = []
                                for user in bucket['get_tiles']:
                                    if self.tile not in bucket['get_tiles'][user][f'{self.chunk}']:
                                        # Pulando chunk não visto
                                        continue
                                    time.append(float(np.average(dectime_val['dectimes'])))
                                    rate.append(float(dectime_val['bitrate']))

                                bucket['seen_tiles'][self.quality]['time'].append(np.average(time))
                                bucket['seen_tiles'][self.quality]['rate'].append(np.average(rate))

                                for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                                    value = quality_val[metric]
                                    idx = self.chunk - 1
                                    avg_qlt = float(np.average(value[idx * 30: idx * 30 + 30]))
                                    if avg_qlt == float('inf'):
                                        avg_qlt = 1000
                                    bucket['seen_tiles'][self.quality][metric].append(avg_qlt)

            print(f' Saving... ', end='')
            save_json(data, data_file)
            del data
            print(f'  Finished.')

        def make_fit(overwrite=False):
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')

            # Load data file
            data = load_json(paths.data_file, object_hook=dict)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        for self.metric in self.metric_list + ['PSNR', 'WS-PSNR', 'S-PSNR']:
                            print(f'  Fitting - {self.proj} {self.tiling} CRF{self.quality} {self.metric}... ', end='')

                            # Check fitter pickle
                            if paths.fitter_pickle_file.exists() and not overwrite:
                                print(f'Pickle found! Skipping.')
                                continue

                            # Load data file
                            samples = data[self.proj][self.tiling][self.quality][self.metric]

                            # Make the fit
                            distributions = self.config['distributions']
                            fitter = Fitter(samples, bins=self.bins, distributions=distributions, timeout=900)
                            fitter.fit()

                            # Saving
                            print(f'  Saving... ', end='')
                            save_pickle(fitter, paths.fitter_pickle_file)
                            print(f'  Finished.')

            del data

        def calc_stats(overwrite=False):
            print('  Calculating Statistics')

            # Check stats file
            if paths.stats_file.exists() and not overwrite:
                warning(f'  stats_file found! Skipping.')
                return

            # Load data file
            data = load_json(paths.data_file)
            stats = defaultdict(list)
            corretations_bucket = defaultdict(list)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    for self.quality in self.quality_list:
                        # Load data
                        data_bucket = data[self.proj][self.tiling]
                        data_bucket = data[self.proj][self.tiling][self.quality]

                        for self.metric in self.metric_list + ['PSNR', 'WS-PSNR', 'S-PSNR']:
                            # Load fitter pickle
                            fitter = load_pickle(paths.fitter_pickle_file)

                            # Calculate percentiles
                            percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
                            df_errors: pd.DataFrame = fitter.df_errors

                            # Calculate errors
                            sse: pd.Series = df_errors['sumsquare_error']
                            bins = len(fitter.x)
                            rmse = np.sqrt(sse / bins)
                            nrmse = rmse / (sse.max() - sse.min())

                            # Append info and stats on Dataframe
                            stats[f'proj'].append(self.proj)
                            stats[f'tiling'].append(self.tiling)
                            stats[f'quality'].append(self.quality)
                            stats[f'metric'].append(self.metric)
                            stats[f'bins'].append(bins)
                            stats[f'average'].append(np.average(data_bucket[self.metric]))
                            stats[f'std'].append(float(np.std(data_bucket[self.metric])))
                            stats[f'min'].append(percentile[0])
                            stats[f'quartile1'].append(percentile[1])
                            stats[f'median'].append(percentile[2])
                            stats[f'quartile3'].append(percentile[3])
                            stats[f'max'].append(percentile[4])

                            # Append distributions on Dataframe
                            for dist in sse.keys():
                                params = fitter.fitted_param[dist]
                                dist_info = self.find_dist(dist, params)

                                stats[f'rmse_{dist}'].append(rmse[dist])
                                stats[f'nrmse_{dist}'].append(nrmse[dist])
                                stats[f'sse_{dist}'].append(sse[dist])
                                stats[f'param_{dist}'].append(dist_info['parameters'])
                                stats[f'loc_{dist}'].append(dist_info['loc'])
                                stats[f'scale_{dist}'].append(dist_info['scale'])

                        for self.metric in self.metric_list + ['PSNR', 'WS-PSNR', 'S-PSNR']:
                            if self.metric != 'time':
                                corr_time_metric = np.corrcoef((data_bucket['time'], data_bucket[self.metric]))[1][0]
                            else:
                                corr_time_metric = 1.0
                            if self.metric != 'rate':
                                corr_rate_metric = np.corrcoef((data_bucket['rate'], data_bucket[self.metric]))[1][0]
                            else:
                                corr_rate_metric = 1.0
                            if self.metric != 'PSNR':
                                corr_psnr_metric = np.corrcoef((data_bucket['PSNR'], data_bucket[self.metric]))[1][0]
                            else:
                                corr_psnr_metric = 1.0
                            if self.metric != 'WS-PSNR':
                                corr_wspsnr_metric = np.corrcoef((data_bucket['WS-PSNR'], data_bucket[self.metric]))[1][0]
                            else:
                                corr_wspsnr_metric = 1.0
                            if self.metric != 'S-PSNR':
                                corr_spsnr_metric = np.corrcoef((data_bucket['S-PSNR'], data_bucket[self.metric]))[1][0]
                            else:
                                corr_spsnr_metric = 1.0

                            corretations_bucket[f'proj'].append(self.proj)
                            corretations_bucket[f'tiling'].append(self.tiling)
                            corretations_bucket[f'metric'].append(self.metric)
                            corretations_bucket[f'corr_time_metric'].append(corr_time_metric)
                            corretations_bucket[f'corr_rate_metric'].append(corr_rate_metric)
                            corretations_bucket[f'corr_psnr_metric'].append(corr_psnr_metric)
                            corretations_bucket[f'corr_wspsnr_metric'].append(corr_wspsnr_metric)
                            corretations_bucket[f'corr_spsnr_metric'].append(corr_spsnr_metric)

            pd.DataFrame(stats).to_csv(str(paths.stats_file), index=False)
            pd.DataFrame(corretations_bucket).to_csv(str(paths.corretations_file), index=False)

        def make_hist(overwrite=False):
            print(f'\n====== Make hist - Bins = {self.bins} ======')

            # Load data
            data = load_json(paths.data_file)

            for self.metric in self.metric_list:
                for self.quality in self.quality_list:
                    # Check image file by metric
                    im_file = self.workfolder / f'hist_tiling_quality_crf{self.quality}_{self.metric}_{self.bins}bins.png'
                    if im_file.exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        continue

                    # Make figure
                    fig = figure.Figure(figsize=(12.8, 3.84))
                    fig2 = figure.Figure(figsize=(12.8, 3.84))  # cdf
                    pos = [(2, 5, x) for x in range(1, 5 * 2 + 1)]
                    subplot_pos = iter(pos)

                    for self.proj in self.proj_list:
                        for self.tiling in self.tiling_list:
                            # Load fitter and select samples
                            fitter = load_pickle(paths.fitter_pickle_file)
                            samples = data[self.proj][self.tiling][self.quality][self.metric]

                            # Position of plot
                            nrows, ncols, index = next(subplot_pos)
                            ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                            ax2: axes.Axes = fig2.add_subplot(nrows, ncols, index)

                            # Make histogram
                            kwrds = dict(label='empirical', color='#dbdbdb', width=fitter.x[1] - fitter.x[0])
                            bins_area = [y * (fitter.x[1] - fitter.x[0]) for y in fitter.y]
                            ax.bar(fitter.x, fitter.y, **kwrds)
                            ax2.bar(fitter.x, np.cumsum(bins_area), **kwrds)

                            # make plot for n_dist distributions
                            dists = fitter.df_errors['sumsquare_error'].sort_values()[0:n_dist].index
                            for dist_name in dists:
                                fitted_pdf = fitter.fitted_pdf[dist_name]

                                dist: scipy.stats.rv_continuous = eval("scipy.stats." + dist_name)
                                param = fitter.fitted_param[dist_name]
                                cdf_fitted = dist.cdf(fitter.x, *param)

                                sse = fitter.df_errors['sumsquare_error'][dist_name]
                                label = f'{dist_name} - SSE {sse: 0.3e}'
                                ax.plot(fitter.x, fitted_pdf, color=self.color_list[dist_name], linewidth=1, label=label)
                                ax2.plot(fitter.x, cdf_fitted, color=self.color_list[dist_name], linewidth=1, label=label)

                            scilimits = (-3, -3) if self.metric == 'time' else (6, 6)
                            title = f'{self.proj.upper()}-{self.tiling}'
                            xlabel = 'Decoding time (ms)' if self.metric == 'time' else 'Bit Rate (Mbps)'
                            ylabel = 'Density' if index in [1, 6] else None
                            ylabel2 = 'Cumulative' if index in [1, 6] else None

                            ax.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
                            ax2.ticklabel_format(axis='x', style='scientific', scilimits=scilimits)
                            ax2.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

                            ax.set_title(title)
                            ax2.set_title(title)
                            ax.set_xlabel(xlabel)
                            ax2.set_xlabel(xlabel)
                            # ax.set_ylabel(ylabel)
                            # ax2.set_ylabel(ylabel2)
                            # ax.set_yscale('log')
                            ax.legend(loc='upper right')
                            ax2.legend(loc='lower right')

                    print(f'  Saving the figure')
                    fig.savefig(im_file)
                    fig2.savefig(im_file.with_stem(f'{im_file.stem}_cdf'))

        def make_bar_tiling_quality(overwrite=False):
            """
            fig = metric
            axes = tiling
            bar = quality
            """
            print(f'\n====== Make bar_tiling_quality ======')

            path = self.workfolder / f'bar_tiling_quality.png'
            if path.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            # Make figure
            figsize = (12.8, 3.84)
            pos = [(2, 5, x) for x in range(1, 2 * 5 + 1)]
            scilimits_y = {'time': (-3, -3), 'rate': (6, 6)}
            xlabel = 'CRF'
            ylabel = {'time': 'Decoding time (ms)', 'rate': 'Bit Rate (Mbps)'}
            x = xticks = list(range(len(self.quality_list)))
            xticklabels = self.quality_list
            colors = {'time': 'tab:blue', 'rate': 'tab:red'}
            legend_handles = [mpatches.Patch(color=colors['time'], label='Time'),
                              mlines.Line2D([], [], color=colors['rate'], label='Bitrate')]

            stats = pd.read_csv(paths.stats_file)

            fig = figure.Figure(figsize=figsize)
            subplot_pos = iter(pos)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    nrows, ncols, index = next(subplot_pos)

                    # Bar plot of dectime
                    data = stats[(stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'time')]
                    height = data[f'average']
                    yerr = data[f'std']

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.bar(x, height, width=0.8, yerr=yerr, color=colors['time'])
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y['time'])

                    # Config labels
                    title = f'{self.proj.upper()} - {self.tiling}'
                    ax.set_title(title)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(xlabel)
                    if index in [1, 6]: ax.set_ylabel(ylabel['time'])
                    ax.legend(handles=legend_handles, loc='upper right')

                    ### Line plot of bit rate
                    data = stats[(stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'rate')]
                    rate_avg = data['average']
                    rate_stdp = rate_avg - data['std']
                    rate_stdm = rate_avg + data['std']

                    ax: axes.Axes = ax.twinx()
                    ax.plot(x, rate_avg, color=colors['rate'], linewidth=1)
                    ax.plot(x, rate_stdp, color='gray', linewidth=1)
                    ax.plot(x, rate_stdm, color='gray', linewidth=1)
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y['rate'])
                    if index in [5, 10]: ax.set_ylabel(ylabel['rate'])

                print(f'Salvando a figura')
                fig.savefig(path)

        def make_bar_quality_tiling(overwrite=False):
            # fig = metric
            # axes = quality
            # bar = tiling
            print(f'\n====== Make bar_quality_tiling ======')

            path = self.workfolder / f'bar_quality_tiling.png'
            if path.exists() and not overwrite:
                warning(f'Figure {path} exist. Skipping')
                return

            figsize = (15.36, 3.84)
            pos = [(2, 6, x) for x in range(1, 2 * 6 + 1)]
            scilimits_y = {'time': (-3, -3), 'rate': (6, 6)}
            xlabel = 'Tiling'
            ylabel = {'time': 'Decoding time (ms)', 'rate': 'Bit Rate (Mbps)'}
            x = xticks = list(range(len(self.tiling_list)))
            xticklabels = self.tiling_list
            legend_handles = [mpatches.Patch(color='tab:blue', label='Time'),
                              mlines.Line2D([], [], color='tab:red', label='Bitrate')]

            stats = pd.read_csv(paths.stats_file)

            fig = figure.Figure(figsize=figsize)
            subplot_pos = iter(pos)

            for self.proj in self.proj_list:
                for self.quality in self.quality_list:
                    nrows, ncols, index = next(subplot_pos)

                    ### Bar plot of dectime
                    data = stats[(stats['quality'] == int(self.quality)) & (stats['proj'] == self.proj) & (stats['metric'] == 'time')]
                    height = data[f'average']
                    yerr = data[f'std']

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.bar(x, height, width=0.8, yerr=yerr, color='tab:blue')
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y['time'])

                    # Config labels
                    title = f'{self.proj.upper()} - CRF {self.quality}'
                    ax.set_title(title)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(xlabel)
                    if index in [1, 7]: ax.set_ylabel(ylabel['time'])
                    ax.legend(handles=legend_handles, loc='upper right')

                    ### Line plot of bit rate
                    data = stats[(stats['quality'] == int(self.quality)) & (stats['proj'] == self.proj) & (stats['metric'] == 'rate')]
                    rate_avg = data['average']
                    rate_stdp = rate_avg - data['std']
                    rate_stdm = rate_avg + data['std']

                    ax: axes.Axes = ax.twinx()
                    ax.plot(x, rate_avg, color='tab:red', linewidth=1)
                    ax.plot(x, rate_stdp, color='gray', linewidth=1)
                    ax.plot(x, rate_stdm, color='gray', linewidth=1)
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y['rate'])
                    if index in [6, 12]:
                        ax.set_ylabel(ylabel['rate'])

                print(f'Salvando a figura')
                fig.savefig(path)

        def make_boxplot(overwrite=False):
            """
            fig = metric
            axes = tiling
            bar = quality & proj
            """
            print(f'\n====== Make boxplot ======')

            # Check image file by metric
            img_file = self.workfolder / f'Boxplot_pattern_quality.png'
            if img_file.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            # Load data
            data = load_json(paths.data_file)

            bar_by_ticks = len(self.proj_list) + 1
            n_ticks = len(self.quality_list)
            bar_colors = {'erp': 'tab:blue', 'cmp': 'tab:red'}
            positions = lambda bar_id: [x for x in range(bar_id, n_ticks * bar_by_ticks, bar_by_ticks)]

            title = "'{} - {}'.format(self.metric.upper(), self.tiling)"
            xlabel = 'CRF'
            xticks = [x + 0.5 for x in positions(0)]
            xticklabels = self.quality_list
            scilimits_y = {'time': (-3, -3), 'rate': (6, 6)}
            ylabel = {'time': 'Decoding time (ms)', 'rate': 'Bit Rate (Mbps)'}
            legend_handles = [mpatches.Patch(color=bar_colors['cmp'], label='CMP'),
                              mpatches.Patch(color=bar_colors['erp'], label='ERP')]

            # Make figure
            figsize = (12.8, 3.84)
            pos = [(2, 5, x) for x in range(1, 2 * 5 + 1)]
            fig = figure.Figure(figsize=figsize)
            subplot_pos = iter(pos)

            fig_boxplot_separated = figure.Figure(figsize=(15.4, 3.84))
            outer = gridspec.GridSpec(2, 5,
                                      #                           # wspace=0.1, hspace=0.1
                                      )
            box_positions = {'erp': 2, 'cmp': 1}

            # make an axes for each metric
            for self.metric in self.metric_list:
                for self.tiling in self.tiling_list:
                    ################ normal axes ################
                    nrows, ncols, index = next(subplot_pos)
                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                    for bar_offset, self.proj in enumerate(self.proj_list):
                        # Group data by quality
                        data_bucket = []
                        for self.quality in self.quality_list:
                            samples = data[self.proj][self.tiling][self.quality][self.metric]
                            data_bucket.append(samples)

                        # Make boxplot
                        artist = ax.boxplot(data_bucket, positions=positions(bar_offset),
                                            widths=1, whis=(0, 100), showfliers=False,
                                            boxprops=dict(facecolor=bar_colors[self.proj]),
                                            flierprops=dict(color='r'),
                                            medianprops=dict(color='k'),
                                            patch_artist=True)
                        for cap in artist['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))
                        # for idx, xtic, line in enumerate(zip(xticks, artist['fliers'])):
                        #     len_fliers = len(line.get_xdata())
                        #     len_total = len(data_bucket[idx])
                        #     y = line.get_xdata()[0] + 0.01 * line.get_xdata()[0]
                        #     x = artist['caps'][idx].get_xdata()[0]
                        #     ax_noliers.text(x, y, f'{len_fliers/len_total:.2f}%',
                        #                     horizontalalignment='center', verticalalignment='center')

                    # Finish normal labels
                    ax.set_title(eval(title))
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(xlabel)
                    ax.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                    if index in [1, 6]: ax.set_ylabel(ylabel['time'])
                    ax.legend(handles=legend_handles, loc='upper right')

                    ################ separated axes ################
                    inner = gridspec.GridSpecFromSubplotSpec(1, len(self.quality_list), subplot_spec=outer[index - 1], )
                    # wspace=0.05, hspace=0.05)
                    for sub_idx, self.quality in enumerate(self.quality_list):
                        ax_sep: axes.Axes = fig_boxplot_separated.add_subplot(inner[sub_idx])

                        for self.proj in self.proj_list:
                            tiling_data = data[self.proj][self.tiling][self.quality][self.metric]
                            boxplot_sep = ax_sep.boxplot((tiling_data,), positions=(box_positions[self.proj],), widths=0.8,
                                                         whis=(0, 100), showfliers=False,
                                                         boxprops=dict(facecolor=bar_colors[self.proj]),
                                                         flierprops=dict(color='r'),
                                                         medianprops=dict(color='k'),
                                                         patch_artist=True)
                            for cap in boxplot_sep['caps']: cap.set_xdata(cap.get_xdata() + (0.3, -0.3))

                        ax_sep.set_title(f'{self.metric.capitalize()}')
                        ax_sep.set_xticks([1.5])
                        ax_sep.set_xticklabels([xticklabels[sub_idx]])
                        if sub_idx in [0] and index in [1]:
                            ax_sep.ticklabel_format(axis='y', style='scientific', scilimits=scilimits_y[self.metric])
                            ax_sep.set_ylabel(ylabel[self.metric])

                        if sub_idx not in [1]:
                            ax_sep.set_yticklabels([])

                        if sub_idx in [5] and index in [5]:
                            ax_sep.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')

            print(f'  Saving the figure')
            fig.savefig(img_file)
            fig_boxplot_separated.savefig(img_file.with_stem(img_file.stem + '_sep'))

        main()

    def by_pattern_full_frame(self, overwrite):
        class ProjectPaths:
            ctx = self

            @property
            def quality_file(self) -> Path:
                quality_file = self.ctx.video_context.project_path / self.ctx.video_context.quality_folder / f'quality_{self.ctx.video}.json'
                return quality_file

            @property
            def dectime_file(self) -> Path:
                dectime_file = self.ctx.video_context.project_path / self.ctx.video_context.dectime_folder / f'dectime_{self.ctx.video}.json'
                return dectime_file

            @property
            def data_file(self) -> Path:
                data_file = self.ctx.workfolder_data / f'data.json'
                return data_file

            @property
            def fitter_pickle_file(self) -> Path:
                fitter_file = self.ctx.workfolder_data / f'fitter_{self.ctx.proj}_{self.ctx.tiling}_{self.ctx.metric}_{self.ctx.bins}bins.pickle'
                return fitter_file

            @property
            def stats_file(self) -> Path:
                stats_file = self.ctx.workfolder / f'stats_{self.ctx.bins}bins.csv'
                return stats_file

            @property
            def corretations_file(self) -> Path:
                corretations_file = self.ctx.workfolder / f'correlations.csv'
                return corretations_file

            @property
            def boxplot_pattern_full_frame_file(self) -> Path:
                img_file = self.ctx.workfolder / f'boxplot_pattern_full_frame_{self.ctx.tiling}_{self.ctx.metric}.png'
                return img_file

            @property
            def hist_pattern_full_frame_file(self) -> Path:
                img_file = self.ctx.workfolder / f'hist_pattern_full_frame_{self.ctx.metric}_{self.ctx.bins}bins.png'
                return img_file

            @property
            def bar_pattern_full_frame_file(self) -> Path:
                img_file = self.ctx.workfolder / f'bar_pattern_full_frame_{self.ctx.metric}.png'
                return img_file

        paths = ProjectPaths()

        def main():
            get_data()
            make_fit()
            calc_stats()
            make_hist()
            make_bar()
            make_boxplot()

        def get_data():
            print('\n\n====== Get Data ======')
            # data[self.proj][self.tiling][self.metric]
            data_file = paths.data_file

            if paths.data_file.exists() and not overwrite:
                warning(f'\n  The data file "{ProjectPaths.data_file}" exist. Loading date.')
                return

            data = AutoDict()

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    for self.metric in ['time', 'time_std', 'rate']:
                        print(f'\r  Getting - {self.proj} {self.tiling} {self.metric}... ', end='')

                        bulcket = data[self.proj][self.tiling][self.metric] = []

                        for self.video in self.config.videos_list:
                            if self.proj not in self.video: continue
                            with self.dectime_ctx() as dectime:
                                for self.quality in self.quality_list:
                                    for self.chunk in self.chunk_list:
                                        # values["time"|"time_std"|"rate"]: list[float|int]

                                        total = 0
                                        for self.tile in self.tile_list:
                                            values = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                            if self.metric == 'time':
                                                total += np.average(values['dectimes'])
                                            elif self.metric == 'rate':
                                                total += np.average(values['bitrate'])
                                        bulcket.append(total)

            print(f' Saving... ', end='')
            save_json(data, paths.data_file)
            del (data)
            print(f'  Finished.')

        def make_fit():
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')

            # Load data file
            data = load_json(paths.data_file)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    for self.metric in self.metric_list:
                        print(f'  Fitting - {self.proj} {self.tiling} {self.metric}... ', end='')

                        # Check fitter pickle
                        if paths.fitter_pickle_file.exists and not overwrite:
                            print(f'Pickle found! Skipping.')
                            continue

                        # Load data file
                        samples = data[self.proj][self.tiling][self.metric]

                        # Calculate bins
                        bins = self.bins
                        if self.bins == 'custom':
                            min_ = np.min(samples)
                            max_ = np.max(samples)
                            norm = round((max_ - min_) / 0.001)
                            if norm > 30:
                                bins = 30
                            else:
                                bins = norm

                        # Make the fit
                        distributions = self.config['distributions']
                        fitter = Fitter(samples, bins=bins, distributions=distributions,
                                        timeout=900)
                        fitter.fit()

                        # Saving
                        print(f'  Saving... ')
                        save_pickle(fitter, paths.fitter_pickle_file)
                        del (data)
                        print(f'  Finished.')

        def calc_stats():
            print('  Calculating Statistics')

            # Check stats file
            if paths.stats_file.exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                return

            data = load_json(paths.data_file)
            stats = defaultdict(list)

            for self.proj in self.proj_list:
                for self.tiling in self.tiling_list:
                    data_bucket = {}

                    for self.metric in self.metric_list:
                        # Load data
                        data_bucket[self.metric] = data[self.proj][self.tiling][self.metric]

                        # Load fitter pickle
                        fitter = load_pickle(ProjectPaths.fitter_pickle_file)

                        # Calculate percentiles
                        percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
                        df_errors: pd.DataFrame = fitter.df_errors

                        # Calculate errors
                        sse: pd.Series = df_errors['sumsquare_error']
                        bins = len(fitter.x)
                        rmse = np.sqrt(sse / bins)
                        nrmse = rmse / (sse.max() - sse.min())

                        # Append info and stats on Dataframe
                        stats[f'proj'].append(self.proj)
                        stats[f'tiling'].append(self.tiling)
                        stats[f'metric'].append(self.metric)
                        stats[f'bins'].append(bins)
                        stats[f'average'].append(np.average(data_bucket[self.metric]))
                        stats[f'std'].append(float(np.std(data_bucket[self.metric])))
                        stats[f'min'].append(percentile[0])
                        stats[f'quartile1'].append(percentile[1])
                        stats[f'median'].append(percentile[2])
                        stats[f'quartile3'].append(percentile[3])
                        stats[f'max'].append(percentile[4])

                        # Append distributions on Dataframe
                        for dist in sse.keys():
                            if dist not in fitter.fitted_param and dist == 'rayleigh':
                                fitter.fitted_param[dist] = (0., 0.)
                            params = fitter.fitted_param[dist]
                            dist_info = self.find_dist(dist, params)

                            stats[f'rmse_{dist}'].append(rmse[dist])
                            stats[f'nrmse_{dist}'].append(nrmse[dist])
                            stats[f'sse_{dist}'].append(sse[dist])
                            stats[f'param_{dist}'].append(dist_info['parameters'])
                            stats[f'loc_{dist}'].append(dist_info['loc'])
                            stats[f'scale_{dist}'].append(dist_info['scale'])

                    corr = np.corrcoef((data_bucket['time'], data_bucket['rate']))[1][0]
                    stats[f'correlation'].append(corr)  # for time
                    stats[f'correlation'].append(corr)  # for rate

            pd.DataFrame(stats).to_csv(str(paths.stats_file), index=False)

        def make_hist():
            print(f'\n====== Make Plot - Bins = {self.bins} ======')
            n_dist = 3

            color_list = {'burr12': 'tab:blue', 'fatiguelife': 'tab:orange',
                          'gamma': 'tab:green', 'invgauss': 'tab:red',
                          'rayleigh': 'tab:purple', 'lognorm': 'tab:brown',
                          'genpareto': 'tab:pink', 'pareto': 'tab:gray',
                          'halfnorm': 'tab:olive', 'expon': 'tab:cyan'}

            # Load data
            data = load_json(paths.data_file)

            # make an image for each metric
            for self.metric in self.metric_list:
                # Check image file by metric
                if paths.hist_pattern_full_frame_file.exists() and not overwrite:
                    warning(f'Figure exist. Skipping')
                    continue

                # Make figure
                fig = figure.Figure(figsize=(12.8, 3.84))
                pos = [(2, 5, x) for x in range(1, 5 * 2 + 1)]
                subplot_pos = iter(pos)

                if self.metric == 'time':
                    xlabel = 'Decoding time (s)'
                    # scilimits = (-3, -3)
                else:
                    xlabel = 'Bit Rate (Mbps)'
                    # scilimits = (6, 6)

                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        # Load fitter and select samples
                        fitter = load_pickle(ProjectPaths.fitter_pickle_file)
                        samples = data[self.proj][self.tiling][self.metric]

                        # Position of plot
                        nrows, ncols, index = next(subplot_pos)
                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                        # Make the histogram
                        self.make_graph('hist', ax, y=samples, bins=len(fitter.x),
                                        label='empirical', title=f'{self.proj.upper()}-{self.tiling}',
                                        xlabel=xlabel)

                        # make plot for n_dist distributions
                        dists = fitter.df_errors['sumsquare_error'].sort_values()[0:n_dist].index
                        for dist_name in dists:
                            fitted_pdf = fitter.fitted_pdf[dist_name]
                            self.make_graph('plot', ax, x=fitter.x, y=fitted_pdf,
                                            label=f'{dist_name}',
                                            color=color_list[dist_name])

                        # ax.set_yscale('log')
                        ax.legend(loc='upper right')

                print(f'  Saving the figure')
                fig.savefig(paths.hist_pattern_full_frame_file)

        def make_bar():
            print(f'\n====== Make Bar - Bins = {self.bins} ======')

            path = paths.bar_pattern_full_frame_file
            if path.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            stats = pd.read_csv(paths.stats_file)
            fig = figure.Figure(figsize=(6.4, 3.84))
            pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
            subplot_pos = iter(pos)

            for self.metric in self.metric_list:
                nrows, ncols, index = next(subplot_pos)
                ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                for start, self.proj in enumerate(self.proj_list):
                    data = stats[(stats[f'proj'] == self.proj) & (stats['metric'] == self.metric)]
                    data_avg = data[f'average']
                    data_std = data[f'std']

                    if self.metric == 'time':
                        ylabel = 'Decoding time (ms)'
                        scilimits = (-3, -3)
                        ax.set_ylim(0.230, 1.250)
                    else:
                        ylabel = 'Bit Rate (Mbps)'
                        scilimits = (6, 6)

                    if self.proj == 'cmp':
                        color = 'tab:green'
                    else:
                        color = 'tab:blue'

                    x = list(range(0 + start, len(data[f'tiling']) * 3 + start, 3))

                    self.make_graph('bar', ax=ax,
                                    x=x, y=data_avg, yerr=data_std,
                                    color=color,
                                    ylabel=ylabel,
                                    title=f'{self.metric}',
                                    scilimits=scilimits)

                # finishing of Graphs
                patch1 = mpatches.Patch(color='tab:green', label='CMP')
                patch2 = mpatches.Patch(color='tab:blue', label='ERP')
                legend = {'handles': (patch1, patch2), 'loc': 'upper right'}
                ax.legend(**legend)
                ax.set_xticks([i + 0.5 for i in range(0, len(self.tiling_list) * 3, 3)])
                ax.set_xticklabels(self.tiling_list)

            print(f'Salvando a figura')
            fig.savefig(path)

        def make_boxplot():
            print(f'\n====== Make Bar - Bins = {self.bins} ======')

            path = paths.boxplot_pattern_full_frame_file
            if path.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            data = load_json(paths.data_file)

            fig = figure.Figure(figsize=(6.4, 3.84))
            pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
            subplot_pos = iter(pos)

            for self.metric in self.metric_list:
                nrows, ncols, index = next(subplot_pos)
                ax: axes.Axes = fig.add_subplot(nrows, ncols, index)

                for start, self.proj in enumerate(self.proj_list):
                    data_bucket = []

                    for self.tiling in self.tiling_list:
                        data_bucket.append(data[self.proj][self.tiling][self.metric])

                    if self.metric == 'time':
                        ylabel = 'Decoding time (ms)'
                        scilimits = (-3, -3)
                    else:
                        ylabel = 'Bit Rate (Mbps)'
                        scilimits = (6, 6)

                    if self.proj == 'cmp':
                        color = 'tab:green'
                    else:
                        color = 'tab:blue'

                    x = list(range(0 + start, len(self.tiling_list) * 3 + start, 3))

                    self.make_graph('boxplot', ax=ax, x=x, y=data_bucket,
                                    title=f'{self.proj.upper()}-{self.metric}',
                                    ylabel=ylabel,
                                    scilimits=scilimits,
                                    color=color)
                    patch1 = mpatches.Patch(color=color, label='CMP')
                    patch2 = mpatches.Patch(color=color, label='ERP')
                    legend = {'handles': (patch1, patch2),
                              'loc': 'upper right'}
                    ax.legend(**legend)
                    ax.set_xticks([i + 0.5 for i in range(0, len(self.tiling_list) * 3, 3)])
                    ax.set_xticklabels(self.tiling_list)

            print(f'Salvando a figura')
            fig.savefig(path)

        main()

    def by_video_by_tiling_by_quality(self, overwrite):
        class ProjectPaths:
            @staticmethod
            def data_file() -> Path:
                data_file = self.workfolder_data / f'data.json'
                return data_file

            @staticmethod
            def fitter_pickle_file() -> Path:
                fitter_file = self.workfolder_data / f'fitter_{self.proj}_{self.video}_{self.tiling}_{self.quality}_{self.metric}_{self.bins}bins.pickle'
                return fitter_file

            @staticmethod
            def stats_file() -> Path:
                stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
                return stats_file

            @staticmethod
            def hist_video_quality_file() -> Path:
                img_file = self.workfolder / f'hist_video_pattern_quality_{self.video}_{self.quality}_{self.metric}_{self.bins}bins.png'
                return img_file

            @staticmethod
            def bar_video_pattern_quality_file() -> Path:
                img_file = self.workfolder / f'bar_video_pattern_quality_{self.name}.png'
                return img_file

            @staticmethod
            def bar_tiling_quality_video_file() -> Path:
                # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
                img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}_{self.metric}.png'
                return img_file

            @staticmethod
            def boxplot_tiling_quality_video_file() -> Path:
                # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
                img_file = self.workfolder / f'boxplot_tiling_quality_video_{self.tiling}_{self.metric}.png'
                return img_file

            @staticmethod
            def boxplot_quality_tiling_video_file() -> Path:
                # img_file = self.workfolder / f'bar_tiling_quality_video_{self.tiling}.png'
                img_file = self.workfolder / f'boxplot_quality_tiling_video_{self.quality}_{self.metric}.png'
                return img_file

        def main():
            get_data()
            make_fit()
            calc_stats()
            # make_bar_video_tiling_quality()
            # make_bar_tiling_quality_video()
            # make_boxplot()
            make_boxplot2()

        def get_data():
            print('\n\n====== Get Data ======')

            if ProjectPaths.data_file().exists() and not overwrite:
                print(f'  The data file "{ProjectPaths.data_file()}" exist. Skipping... ', end='')
                return
            data = AutoDict()

            for self.proj in self.proj_list:
                for self.video in self.videos_list:
                    if self.proj not in self.video: continue
                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            for self.metric in ['time', 'time_std', 'rate']:
                                print(f'  Getting -  {self.proj} {self.name} {self.tiling} CRF{self.quality} {self.metric}... ', end='')

                                bucket = data[self.name][self.proj][self.tiling][self.quality][self.metric] = []

                                with self.dectime_ctx() as dectime:
                                    for self.tile in self.tile_list:
                                        for self.chunk in self.chunk_list:
                                            # values["time"|"time_std"|"rate"]: list[float|int]
                                            values = dectime[self.tiling][self.quality][f'{self.tile}'][f'{self.chunk}']
                                            bucket.append(np.average(values[self.metric]))
                                print(f'  OK.')

            print(f' Saving... ', end='')

            save_json(data, ProjectPaths.data_file())
            print(f'  Finished.')

        def make_fit():
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')

            # Load data file
            data = load_json(ProjectPaths.data_file())

            for self.video in self.videos_list:
                for self.proj in self.proj_list:
                    if self.proj not in self.video: continue
                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            for self.metric in self.metric_list:
                                print(f'  Fitting - {self.proj} {self.video} {self.tiling} CRF{self.quality} {self.metric}... ', end='')

                                # Check fitter pickle
                                if ProjectPaths.fitter_pickle_file().exists() and not overwrite:
                                    print(f'Pickle found! Skipping.')
                                    continue

                                # Calculate bins
                                bins = self.bins
                                if self.bins == 'custom':
                                    min_ = np.min(data)
                                    max_ = np.max(data)
                                    norm = round((max_ - min_) / 0.001)
                                    if norm > 30:
                                        bins = 30
                                    else:
                                        bins = norm

                                # Make the fit
                                samples = data[self.name][self.proj][self.tiling][self.quality][self.metric]
                                distributions = self.config['distributions']
                                ft = Fitter(samples, bins=bins, distributions=distributions,
                                            timeout=900)
                                ft.fit()

                                # Saving
                                print(f'  Saving... ', end='')
                                save_pickle(ft, ProjectPaths.fitter_pickle_file())

                                print(f'  Finished.')
            del data

        def calc_stats():
            print(f'\n\n====== Make Statistics - Bins = {self.bins} ======')
            # Check stats file
            if ProjectPaths.stats_file().exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                return

            data = load_json(ProjectPaths.data_file())
            stats = defaultdict(list)
            for self.proj in self.proj_list:
                for self.video in self.videos_list:
                    if self.proj not in self.video: continue
                    for self.tiling in self.tiling_list:
                        for self.quality in self.quality_list:
                            data_bucket = {}

                            for self.metric in self.metric_list:
                                # Load data file
                                data_bucket[self.metric] = data[self.name][self.proj][self.tiling][self.quality][self.metric]

                                # Load fitter pickle
                                fitter = load_pickle(ProjectPaths.fitter_pickle_file())

                                # Calculate percentiles
                                percentile = np.percentile(data_bucket[self.metric], [0, 25, 50, 75, 100]).T
                                df_errors: pd.DataFrame = fitter.df_errors

                                # Calculate errors
                                sse: pd.Series = df_errors['sumsquare_error']
                                bins = len(fitter.x)
                                rmse = np.sqrt(sse / bins)
                                nrmse = rmse / (sse.max() - sse.min())

                                # Append info and stats on Dataframe
                                stats[f'proj'].append(self.proj)
                                stats[f'video'].append(self.video)
                                stats[f'name'].append(self.name)
                                stats[f'tiling'].append(self.tiling)
                                stats[f'quality'].append(self.quality)
                                stats[f'metric'].append(self.metric)
                                stats[f'bins'].append(bins)
                                stats[f'average'].append(np.average(data_bucket[self.metric]))
                                stats[f'std'].append(float(np.std(data_bucket[self.metric])))
                                stats[f'min'].append(percentile[0])
                                stats[f'quartile1'].append(percentile[1])
                                stats[f'median'].append(percentile[2])
                                stats[f'quartile3'].append(percentile[3])
                                stats[f'max'].append(percentile[4])

                                # Append distributions on Dataframe
                                for dist in sse.keys():
                                    if dist not in fitter.fitted_param and dist == 'rayleigh':
                                        fitter.fitted_param[dist] = (0., 0.)
                                    params = fitter.fitted_param[dist]
                                    dist_info = self.find_dist(dist, params)

                                    stats[f'rmse_{dist}'].append(rmse[dist])
                                    stats[f'nrmse_{dist}'].append(nrmse[dist])
                                    stats[f'sse_{dist}'].append(sse[dist])
                                    stats[f'param_{dist}'].append(dist_info['parameters'])
                                    stats[f'loc_{dist}'].append(dist_info['loc'])
                                    stats[f'scale_{dist}'].append(dist_info['scale'])

                            corr = np.corrcoef((data_bucket['time'], data_bucket['rate']))[1][0]
                            stats[f'correlation'].append(corr)  # for time
                            stats[f'correlation'].append(corr)  # for rate

            pd.DataFrame(stats).to_csv(str(ProjectPaths.stats_file()), index=False)

        def make_bar_video_tiling_quality():
            print(f'\n====== Make Bar1 - Bins = {self.bins} ======')
            stats = pd.read_csv(ProjectPaths.stats_file())

            for self.video in stats['name'].unique():
                if ProjectPaths.bar_video_pattern_quality_file().exists() and not overwrite:
                    warning(f'Figure exist. Skipping')
                    return

                fig = figure.Figure()
                subplot_pos = iter(((2, 5, 1), (2, 5, 2), (2, 5, 3), (2, 5, 4), (2, 5, 5), (2, 5, 6), (2, 5, 7), (2, 5, 8), (2, 5, 9), (2, 5, 10)))
                for self.proj in self.proj_list:
                    for self.tiling in self.tiling_list:
                        nrows, ncols, index = next(subplot_pos)
                        ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                        time_stats = stats[(stats['name'] == self.video) & (stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'time')]
                        x = xticks = time_stats['quality']
                        time_avg = time_stats['average']
                        time_std = time_stats['std']
                        ylabel = 'Decoding time (s)' if index in [1, 6] else None

                        self.make_graph('bar', ax=ax,
                                        x=x, y=time_avg, yerr=time_std,
                                        title=f'{self.proj}-{self.tiling}',
                                        xlabel='CRF',
                                        ylabel=ylabel,
                                        xticks=xticks,
                                        width=5,
                                        scilimits=(-3, -3))

                        # line plot of bit rate
                        ax = ax.twinx()

                        rate_stats = stats[(stats['name'] == self.video) & (stats['tiling'] == self.tiling) & (stats['proj'] == self.proj) & (stats['metric'] == 'rate')]
                        rate_avg = rate_stats['average']
                        rate_stdp = rate_avg - rate_stats['std']
                        rate_stdm = rate_avg + rate_stats['std']
                        legend = {'handles': (mpatches.Patch(color='#1f77b4', label='Time'),
                                              mlines.Line2D([], [], color='red', label='Bitrate')),
                                  'loc': 'upper right'}
                        ylabel = 'Bit Rate (Mbps)' if index in [5, 10] else None

                        self.make_graph('plot', ax=ax,
                                        x=x, y=rate_avg,
                                        legend=legend,
                                        ylabel=ylabel,
                                        color='r',
                                        scilimits=(6, 6))

                        ax.plot(x, rate_stdp, color='gray', linewidth=1)
                        ax.plot(x, rate_stdm, color='gray', linewidth=1)
                        ax.ticklabel_format(axis='y', style='scientific',
                                            scilimits=(6, 6))
                # fig.show()
                print(f'Salvando a figura')

                fig.savefig(ProjectPaths.bar_video_pattern_quality_file())

        def make_bar_tiling_quality_video():
            print(f'\n====== Make Bar2 - Bins = {self.bins} ======')
            stats_df = pd.read_csv(ProjectPaths.stats_file())

            for self.tiling in self.tiling_list:
                # pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
                # subplot_pos = iter(pos)
                # fig = figure.Figure(figsize=(12, 4), dpi=300)

                for self.metric in ['time', 'rate']:
                    if ProjectPaths.bar_tiling_quality_video_file().exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        return

                    pos = [(1, 1, 1)]
                    subplot_pos = iter(pos)
                    fig = figure.Figure(figsize=(12, 4), dpi=300)

                    nrows, ncols, index = next(subplot_pos)
                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    xticks = stats_df['name'].unique()
                    start = -1
                    for self.quality in self.quality_list:
                        for self.proj in self.proj_list:
                            start += 1
                            stats = stats_df[(stats_df['quality'] == int(self.quality)) & (stats_df['tiling'] == self.tiling) & (stats_df['proj'] == self.proj) & (
                                        stats_df['metric'] == self.metric)]
                            stats = stats.sort_values(by=['average'], ascending=False)

                            x = list(range(0 + start, len(xticks) * 13 + start, 13))
                            time_avg = stats['average']
                            time_std = stats['std']

                            if self.metric == 'time':
                                ylabel = 'Decoding time (s)'
                                scilimits = (-3, -3)
                            else:
                                ylabel = 'Bit Rate (Mbps)'
                                scilimits = (6, 6)

                            if self.proj == 'cmp':
                                color = 'lime'
                            else:
                                color = 'blue'
                            self.make_graph('bar', ax=ax,
                                            x=x, y=time_avg, yerr=time_std,
                                            title=f'{self.tiling}-{self.metric}',
                                            ylabel=ylabel,
                                            width=1,
                                            color=color,
                                            scilimits=scilimits)
                    ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
                    ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
                    print(f'Salvando a figura')
                    fig.savefig(ProjectPaths.bar_tiling_quality_video_file())

                # print(f'Salvando a figura')
                # fig.savefig(ProjectPaths.bar_tiling_quality_video_file())

        def make_boxplot():
            print(f'\n====== Make Boxplot - Bins = {self.bins} ======')
            stats_df = pd.read_csv(ProjectPaths.stats_file())
            data = load_json(ProjectPaths.data_file())

            for self.tiling in self.tiling_list:
                for self.metric in self.metric_list:
                    if ProjectPaths.boxplot_tiling_quality_video_file().exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        return

                    stats_df = stats_df.sort_values(['average'], ascending=False)

                    pos = [(1, 1, 1)]
                    subplot_pos = iter(pos)
                    fig = figure.Figure(figsize=(20, 8), dpi=300)

                    nrows, ncols, index = next(subplot_pos)
                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.grid(axis='y')
                    xticks = stats_df['name'].unique()
                    start = -1

                    for self.quality in self.quality_list:
                        for self.proj in ['cmp', 'erp']:
                            start += 1
                            data_bucket = []

                            for name in xticks:
                                data_bucket.append(data[name][self.proj][self.tiling][self.quality][self.metric])
                            x = list(range(0 + start, len(xticks) * 13 + start, 13))

                            if self.metric == 'time':
                                ylabel = 'Decoding time (s)'
                                scilimits = (-3, -3)
                            else:
                                ylabel = 'Bit Rate (Mbps)'
                                scilimits = (6, 6)

                            if self.proj == 'cmp':
                                color = 'lime'
                            else:
                                color = 'blue'

                            self.make_graph('boxplot', ax=ax,
                                            x=x, y=data_bucket,
                                            title=f'{self.tiling}-{self.metric}',
                                            ylabel=ylabel,
                                            width=1,
                                            color=color,
                                            scilimits=scilimits)
                            patch1 = mpatches.Patch(color='lime', label='CMP')
                            patch2 = mpatches.Patch(color='blue', label='ERP')
                            legend = {'handles': (patch1, patch2),
                                      'loc': 'upper right'}
                            ax.legend(**legend)

                    ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
                    ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
                    print(f'Salvando a figura')
                    fig.savefig(ProjectPaths.boxplot_tiling_quality_video_file())

        def make_boxplot2():
            print(f'\n====== Make Boxplot2 - Bins = {self.bins} ======')
            stats_df = pd.read_csv(ProjectPaths.stats_file())
            data = load_json(ProjectPaths.data_file())

            for self.quality in self.quality_list:
                # pos = [(2, 1, x) for x in range(1, 2 * 1 + 1)]
                # subplot_pos = iter(pos)
                # fig = figure.Figure(figsize=(12, 4), dpi=300)

                for self.metric in ['time', 'rate']:
                    if ProjectPaths.boxplot_quality_tiling_video_file().exists() and not overwrite:
                        warning(f'Figure exist. Skipping')
                        return

                    stats_df = stats_df.sort_values(['average'], ascending=False)

                    pos = [(1, 1, 1)]
                    subplot_pos = iter(pos)
                    fig = figure.Figure(figsize=(20, 8), dpi=300)

                    nrows, ncols, index = next(subplot_pos)
                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.grid(axis='y')
                    xticks = stats_df['name'].unique()
                    start = -1
                    for self.tiling in self.tiling_list:
                        for self.proj in ['cmp', 'erp']:
                            start += 1
                            data_bucket = []
                            for name in xticks:
                                data_bucket.append(data[name][self.proj][self.tiling][self.quality][self.metric])
                            x = list(range(0 + start, len(xticks) * 13 + start, 13))

                            if self.metric == 'time':
                                ylabel = 'Decoding time (s)'
                                scilimits = (-3, -3)
                            else:
                                ylabel = 'Bit Rate (Mbps)'
                                scilimits = (6, 6)

                            if self.proj == 'cmp':
                                color = 'lime'
                            else:
                                color = 'blue'

                            self.make_graph('boxplot', ax=ax,
                                            x=x, y=data_bucket,
                                            title=f'CRF{self.quality}-{self.metric}',
                                            ylabel=ylabel,
                                            width=1,
                                            color=color,
                                            scilimits=scilimits)
                            patch1 = mpatches.Patch(color='lime', label='CMP')
                            patch2 = mpatches.Patch(color='blue', label='ERP')
                            legend = {'handles': (patch1, patch2),
                                      'loc': 'upper right'}
                            ax.legend(**legend)

                    ax.set_xticks(list(range(0 + 6, len(xticks) * 13 + 6, 13)))
                    ax.set_xticklabels(xticks, rotation=13, ha='right', va='top')
                    print(f'Salvando a figura')
                    fig.savefig(ProjectPaths.boxplot_quality_tiling_video_file())

        main()


class Dashing(BaseTileDecodeBenchmark):
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


class CheckTiles(BaseTileDecodeBenchmark):
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

    def check_decode(self, only_error=True, clean=False):
        dectime_log = self.video_context.dectime_log
        debug(f'Checking the file {dectime_log}')

        if not dectime_log.exists():
            warning('logfile_not_found')
            count_decode = 0
        else:
            count_decode = self.count_decoding()

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


class QualityMetrics:
    PIXEL_MAX: int = 255
    video_context: VideoContext = None
    weight_ndarray: Union[np.ndarray, object] = np.zeros(0)
    sph_points_mask: np.ndarray = np.zeros(0)
    sph_points_img: list = []
    sph_points: list = []
    cart_coord: list = []
    sph_file: Path
    results: AutoDict

    # ### Coordinate system ### #
    # Image coordinate system
    ICSPoint = NamedTuple('ICSPoint', (('x', float), ('y', float)))
    # Horizontal coordinate system
    HCSPoint = NamedTuple('HCSPoint',
                          (('azimuth', float), ('elevation', float)))
    # Aerospacial coordinate system
    ACSPoint = NamedTuple('ACSPoint',
                          (('yaw', float), ('pitch', float), ('roll', float)))
    # Cartesian coordinate system
    CCSPoint = NamedTuple('CCSPoint',
                          (('x', float), ('y', float), ('z', float)))

    # ### util ### #
    def mse2psnr(self, mse: float) -> float:
        return 10 * np.log10((self.PIXEL_MAX ** 2 / mse))

    # ### psnr ### #
    def psnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
             im_sal: np.ndarray = None) -> float:
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
        return self.mse2psnr(mse)

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
    def prepare_weight_ndarray(self):
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

    def wspsnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
               im_sal: np.ndarray = None):
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :param im_sal:
        :return:
        """

        if self.video_context.video.resolution.shape != self.weight_ndarray.shape:
            self.prepare_weight_ndarray()

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

        return self.mse2psnr(wmse)

    # ### spsnr_nn ### #
    def spsnr_nn(self, im_ref: np.ndarray,
                 im_deg: np.ndarray,
                 im_sal: np.ndarray = None):
        """
        Calculate of S-PSNR between two images. All arrays must be on the same
        resolution.

        :param im_ref: The original image
        :param im_deg: The image degraded
        :param im_sal: The saliency map
        :return:
        """
        # sph_file = Path('lib/sphere_655362.txt'),
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

        mse = sqr_dif.sum() / mask.sum()
        return self.mse2psnr(mse)

    def ffmpeg_psnr(self):
        if self.video_context.chunk == 1:
            name, pattern, quality, tile, chunk = self.video_context.state
            results = self.results[name][pattern][str(quality)][tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        psnr = None
        compressed_log = self.video_context.compressed_file.with_suffix('.log')
        content = compressed_log.read_text(encoding='utf-8')
        content = content.splitlines()

        for line in content:
            if 'Global PSNR' in line:
                psnr = {'psnr': get_psnr(line),
                        'qp_avg': get_qp(line)}
                break
        return psnr


class QualityAssessment(BaseTileDecodeBenchmark, QualityMetrics):
    results: AutoDict
    old_video = ''

    def __init__(self, config: str,
                 role: str,
                 sph_file=Path('lib/sphere_655362.txt'),
                 **kwargs):
        """
        Load configuration and run the main routine defined on Role Operation.

        :param config: a Config object
        :param role: The role can be: ALL, PSNR, WSPSNR, SPSNR, RESULTS
        :param sphere_file: a Path for sphere_655362.txt
        :param kwargs: passed to main routine
        """
        operations = {
            'ALL': Role(name='ALL', deep=4, operation=self.all),
            'PSNR': Role(name='PSNR', deep=4, operation=self.only_a_metric),
            'WSPSNR': Role(name='WSPSNR', deep=4, operation=self.only_a_metric),
            'SPSNR': Role(name='SPSNR', deep=4, operation=self.only_a_metric),
            'RESULTS': Role(name='RESULTS', deep=4, operation=self.result,
                            finish=self.save_result),
        }

        self.metrics = {'PSNR': self.psnr,
                        'WS-PSNR': self.wspsnr,
                        'S-PSNR': self.spsnr_nn}

        self.sph_file = sph_file

        self.results_dataframe = pd.DataFrame()

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    def all_init(self):
        pass

    def all(self, overwrite=False):
        debug(f'Processing {self.video_context}')

        if self.video_context.quality == self.video_context.original_quality:
            # info('Skipping original quality')
            return 'continue'

        reference_file = self.video_context.reference_file
        compressed_file = self.video_context.compressed_file
        quality_csv = self.video_context.quality_video_csv

        metrics = self.metrics.copy()

        if quality_csv.exists() and not overwrite:
            csv_dataframe = pd.read_csv(quality_csv, encoding='utf-8',
                                        index_col=0)
            for metric in csv_dataframe:
                debug(
                    f'The metric {metric} exist for {self.video_context.state}. '
                    'Skipping this metric')
                del (metrics[metric])
            if metrics == {}:
                warning(f'The metrics for {self.video_context.state} are OK. '
                        f'Skipping')
                return 'continue'
        else:
            csv_dataframe = pd.DataFrame()

        results = defaultdict(list)

        frames = zip(iter_frame(reference_file), iter_frame(compressed_file))
        start = time.time()
        for n, (frame_video1, frame_video2) in enumerate(frames, 1):
            for metric in metrics:
                metrics_method = self.metrics[metric]
                metric_value = metrics_method(frame_video1, frame_video2)
                results[metric].append(metric_value)
            print(
                f'{self.video_context.state} - Frame {n} - {time.time() - start: 0.3f} s',
                end='\r')

        for metric in results:
            csv_dataframe[metric] = results[metric]

        csv_dataframe.to_csv(quality_csv, encoding='utf-8', index_label='frame')
        print('')

    def only_a_metric(self, **kwargs):
        self.metrics = self.metrics[self.role.name]
        self.all(**kwargs)

    # def init_result(self):
    #     quality_result_json = self.video_context.quality_result_json
    #
    #     if quality_result_json.exists():
    #         warning(f'The file {quality_result_json} exist. Loading.')
    #         json_content = quality_result_json.read_text(encoding='utf-8')
    #         self.results = load_json(json_content)
    #     else:
    #         self.results = AutoDict()

    def result(self, overwrite=False):
        debug(f'Processing {self.video_context}')
        if self.video_context.quality == self.video_context.original_quality:
            # info('Skipping original quality')
            return 'continue'

        results = self.results
        quality_csv = self.video_context.quality_video_csv  # The compressed quality

        if not quality_csv.exists():
            warning(f'The file {quality_csv} not exist. Skipping.')
            return 'continue'

        csv_dataframe = pd.read_csv(quality_csv, encoding='utf-8', index_col=0)

        for key in self.video_context.state:
            results = results[key]

        for metric in self.metrics:
            if results[metric] != {} and not overwrite:
                warning(f'The metric {metric} exist for Result '
                        f'{self.video_context.state}. Skipping this metric')
                return

            try:
                results[metric] = csv_dataframe[metric].tolist()
                if len(results[metric]) == 0:
                    raise KeyError
            except KeyError:
                warning(f'The metric {metric} not exist for csv_dataframe'
                        f'{self.video_context.state}. Skipping this metric')
                return
        if self.old_video != f'{self.video_context.video}':
            self.old_video = f'{self.video_context.video}'
            self.save_result()

        return 'continue'

    def save_result(self):
        quality_result_json = self.video_context.quality_result_json
        save_json(self.results, quality_result_json)

        quality_result_pickle = quality_result_json.with_suffix('.pickle')
        save_pickle(self.results, quality_result_pickle)

    def ffmpeg_psnr(self):
        if self.video_context.chunk == 1:
            name, pattern, quality, tile, chunk = self.video_context.state
            results = self.results[name][pattern][quality][tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        psnr = None
        compressed_log = self.video_context.compressed_file.with_suffix('.log')
        content = compressed_log.read_text(encoding='utf-8')
        content = content.splitlines()

        for line in content:
            if 'Global PSNR' in line:
                psnr = {'psnr': get_psnr(line),
                        'qp_avg': get_qp(line)}
                break
        return psnr


class MakeViewport(BaseTileDecodeBenchmark):
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
                    height, width = erp.proj_res

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


class GetTiles(BaseTileDecodeBenchmark, Graphs):
    results: AutoDict
    database: AutoDict
    old_video = None
    old_tiling = None
    old_chunk = None
    new_database: AutoDict
    get_tiles: AutoDict
    name: str

    def __init__(self, config: str, role: str,
                 **kwargs):
        """
        Load configuration and run the main routine defined on Role Operation.

        :param config: a Config object
        :param role: The role can be: ALL, PSNR, WSPSNR, SPSNR, RESULTS
        :param sphere_file: a Path for sphere_655362.txt
        :param kwargs: passed to main routine
        """
        operations = {
            'PREPARE': Role(name='PREPARE', deep=0,
                            init=None,
                            operation=self.process_nasrabadi,
                            finish=None),
            'GET_TILES': Role(name='GET_TILES', deep=0,
                              init=None,
                              operation=self.get_tiles,
                              finish=None),
            'USER_ANALISYS': Role(name='USER_ANALISYS', deep=0,
                              init=None,
                              operation=self.user_analisys,
                              finish=None),
            }

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)
        self.overwrite = kwargs['overwrite']

        self.dataset_name = kwargs.pop('dataset_name')
        self.database_folder = Path('datasets')
        self.database_path = self.database_folder / self.dataset_name
        self.database_json = self.database_path / f'{self.dataset_name}.json'
        self.database_pickle = self.database_path / f'{self.dataset_name}.pickle'

        self.get_tiles_path = (self.video_context.project_path
                               / self.video_context.get_tiles_folder)
        self.get_tiles_path.mkdir(parents=True, exist_ok=True)

        self.role.operation()

    def process_nasrabadi(self):
        print('Processing dataset.')
        database_json = self.database_json
        rotation_map = {'cable_cam_nas': 265, 'drop_tower_nas': 180,
                        'wingsuit_dubai_nas': 63, 'drone_chases_car_nas': 81}

        if database_json.exists() and not self.overwrite:
            warning(f'The file {database_json} exist. Skipping.')
            return

        database = AutoDict()
        video_id_map = load_json(f'{self.database_path}/videos_map.json', object_hook=dict)
        user_map = {}

        for csv_database_file in self.database_path.glob('*/*.csv'):
            # Parse filename in user_id and video_id
            user_nas_id, video_nas_id = csv_database_file.stem.split('_')
            video_name = video_id_map[video_nas_id]

            # Map user_id to str(int)
            try:
                user_id = str(user_map[user_nas_id])
            except KeyError:
                user_map[user_nas_id] = len(user_map)
                user_id = str(user_map[user_nas_id])

            print(f'\rUser {user_id} - {video_name} - ', end='')

            # "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
            # respectively, to reorient during playback." - Author
            # Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
            rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0

            previous_line = None
            yaw_pitch_roll_frames = []
            start_time = time.time()
            names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
            n = 0
            frame_counter = 0

            # with open(csv_database_file, newline='') as csv_database:
            with open(csv_database_file) as csv_database:
                head_movement = pd.read_csv(csv_database_file, names=names)
                for n, line in enumerate(head_movement.itertuples(index=False, name=None)):
                    # for n, line in enumerate(csv.reader(csv_database)):
                    if len(yaw_pitch_roll_frames) >= 1800: break
                    timestamp, Qx, Qy, Qz, Qw, Vx, Vy, Vz = map(float, line)
                    x, y, z = Vz, Vy, Vx  # Based on gitHub code of author

                    frame_timestamp = frame_counter / 30
                    if timestamp < frame_timestamp:
                        previous_line = timestamp, x, y, z
                        continue

                    if timestamp > frame_timestamp:
                        # Linear Interpolation
                        old_timestamp, old_x, old_y, old_z = previous_line
                        t: float = frame_timestamp
                        t_i: float = old_timestamp
                        t_f: float = timestamp
                        v_i: tuple = old_x, old_y, old_z
                        v_f: tuple = x, y, z
                        x, y, z = lin_interpol(t, (t_f, t_i), (v_f, v_i))

                    azimuth, elevation = cart2hcs(x, y, z)
                    yaw, pitch = map(np.rad2deg, (azimuth, elevation))
                    yaw, pitch = map(round, (yaw, pitch), (3, 3))
                    roll = 0
                    yaw = (yaw + 180.) % 360 - 180
                    if yaw >= 180 or yaw < -180:
                        print(f'\n{frame_counter=}, {yaw=}°')
                    if pitch > 90 or pitch < -90:
                        print(f'\n{frame_counter=}, {pitch=}°')

                    if video_name in rotation_map:
                        yaw -= rotation_map[video_name]
                        yaw = (yaw + 180.) % 360 - 180

                    yaw_pitch_roll_frames.append((yaw, pitch, roll))

                    frame_counter += 1
                    previous_line = timestamp, x, y, z
            while len(yaw_pitch_roll_frames) < 1800:
                yaw_pitch_roll_frames.append(yaw_pitch_roll_frames[-1])
            database[video_name][user_id] = yaw_pitch_roll_frames
            print(f'Samples {n:04d} - {frame_counter=} - {time.time() - start_time:0.3f}')

        print(f'\nFinish. Saving as {database_json}.')
        save_json(database, database_json)

    def get_tiles(self):
        print(f'Get Tiles.')
        self.database = load_json(self.database_json)

        for self.video in self.videos_list:
            users_data = self.database[self.name]

            for self.tiling in self.tiling_list:
                get_tiles_json = self.get_tiles_path / f'get_tiles_{self.dataset_name}_{self.video}_{self.tiling}.json'
                results = AutoDict()

                if get_tiles_json.exists() and not self.overwrite:
                    warning(f'\nThe file {get_tiles_json} exist. Loading.')
                    results = load_json(get_tiles_json)

                # erp '144x72', '288x144','432x216','576x288'
                # cmp '144x96', '288x192','432x288','576x384'
                erp = ERP(f'{self.tiling}', '576x288', '90x90')
                print(f'{self.name} - tiling {self.tiling}')

                for user in users_data:
                    start = time.time()

                    yaw_pitch_roll_frames = users_data[user]
                    result: list[list[int, ...]]
                    result_tiles = []
                    result_counter = Counter()
                    result_chunks = {}

                    chunk = 0
                    tiles_chunks = set()

                    for frame, (yaw, pitch, roll) in enumerate(yaw_pitch_roll_frames):
                        if (frame + 1) % 30 == 0:
                            print(f'\r  {user} - {frame:04d} - ', end='')

                        # vptiles
                        if results[self.vid_proj][self.tiling][user]['tiles'] == {}:
                            yaw, pitch, roll = np.deg2rad((yaw, pitch, roll))
                            erp.set_vp(yaw, pitch, roll)
                            vptiles = erp.get_vptiles()
                            result_tiles.append(vptiles)
                        else:
                            vptiles = results[self.vid_proj][self.tiling][user]['tiles'][frame]

                        # Calcule vptiles by chunk
                        if results[self.vid_proj][self.tiling][user]['chunks'] == {}:
                            tiles_chunks.update(vptiles)
                            if (frame + 1) % 30 == 0:
                                chunk += 1  # start from 1 gpac defined
                                result_chunks[f'{chunk}'] = list(tiles_chunks)
                                result_counter += Counter(tiles_chunks)
                                tiles_chunks = set()

                        if (frame + 1) % 30 == 0:
                            print(f'{time.time() - start:.3f}s - {repr(result_counter)}          ', end='')

                    print('')

                    if results[self.vid_proj][self.tiling][user]['tiles'] == {}:
                        results[self.vid_proj][self.tiling][user]['tiles'] = result_tiles
                    if results[self.vid_proj][self.tiling][user]['counter'] == {}:
                        results[self.vid_proj][self.tiling][user]['counter'] = dict(result_counter)
                    if results[self.vid_proj][self.tiling][user]['chunks'] == {}:
                        results[self.vid_proj][self.tiling][user]['chunks'] = result_chunks

                    # todo: remove this in the future
                    break
                save_json(results, get_tiles_json)

    def user_analisys(self):
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


class GetViewportQuality(BaseTileDecodeBenchmark):
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
