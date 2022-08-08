from __future__ import print_function
import datetime
import json
import pickle
import time
from builtins import PermissionError
from collections import Counter, defaultdict
from logging import warning, debug, fatal, info
from pathlib import Path
from subprocess import run, DEVNULL
from typing import Any, NamedTuple, Union, Dict, Tuple, List, Optional
from cycler import cycler
from fitter import Fitter

import numpy as np
import pandas as pd
import matplotlib.axes as axes
import matplotlib.figure as figure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .assets import AutoDict, Role
from .util import (run_command, check_video_gop, iter_frame, load_sph_file,
                   xyz2hcs, save_json, load_json, lin_interpol, save_pickle,
                   load_pickle)
from .video_state import Config, VideoContext


class BaseTileDecodeBenchmark:
    config: Config = None
    video_context: VideoContext = None
    role: Role = None

    def run(self, **kwargs):
        self.print_resume()
        self.role.init()

        # total = len(self.video_context)
        total = 0
        for n in self.video_context:
            print(f'{n}/{total}', end='\r', flush=True)
            info(f'\n{self.video_context.state}')
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
        f = self.video_context.dectime_log.open('r')
        for line in f:
            time = line.strip().split(' ')[1].split('=')[1][:-1]
            times.append(float(time))
        f.close()
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

        info(f'==== Processing {segment_folder} ====')

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
        info(f'==== Processing {dectime_log} ====')

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

        self.run_times(diff, (cmd, dectime_log, 'a'))

    @staticmethod
    def run_times(num, args):
        for i in range(num):
            run_command(*args)

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
        if dectime_json_file.exists() and not overwrite:
            warning(f'The file {dectime_json_file} exist and not overwrite. Skipping.')
            return 'exit'

        results = self.results
        for factor in self.video_context.state:
            results = results[factor]

        if not results == {} and not overwrite:
            warning(f'The result key for "{self.video_context.state}" contain some value. '
                    f'Skipping.')
            return 'skip'  # if value exist and not overwrite, then skip

        if not self.video_context.segment_file.exists():
            warning(f'The file {self.video_context.segment_file} not exist. Skipping.')
            return 'skip'

        if not self.video_context.dectime_log.exists():
            warning(f'The file {self.video_context.dectime_log} not exist. Skipping.')
            return 'skip'

        try:
            chunk_size = self.video_context.segment_file.stat().st_size
            bitrate = 8 * chunk_size / self.video_context.video.chunk_dur
        except PermissionError:
            warning(f'PermissionError error on reading size of '
                    f'{self.video_context.segment_file}. Skipping.')
            return 'skip'

        dectimes: List[float] = self.get_times()

        data = {'dectimes': dectimes, 'bitrate': bitrate}
        print(f'\r{data}', end='')

        results.update(data)

    def save_dectime(self):
        filename = self.video_context.dectime_json_file
        info(f'Saving {filename}')
        save_json(self.results, filename)
        filename = self.video_context.dectime_json_file.with_suffix('.pickle')
        save_pickle(self.results, filename)


class DectimeGraphs(BaseTileDecodeBenchmark):
    # Fitter
    fit: Fitter
    fitter = AutoDict()
    fit_errors = AutoDict()

    stats: pd.DataFrame
    data = AutoDict()
    data.__doc__ = '[tiling]["time"|"time_std"|"rate"]'

    name = tiling = quality = tile = chunk = None
    dectime_filename: Path
    dectime: AutoDict
    data_file: Path
    plot_config: AutoDict
    config: Config
    role: Role
    video_context: VideoContext
    workfolder: Path
    workfolder_data: Path
    bins: Union[int, str]
    color_list: Tuple[str]
    subplot_pos: Tuple[Tuple[int, int, int]]

    def __init__(self, config: str, role: str,
                 role_folder: str,
                 bins: List[Union[str, int]],
                 **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        operations = {
            'HIST_BY_PATTERN': Role(name='HIST_BY_PATTERN', deep=0,
                                    init=self.init_hist_by_pattern,
                                    operation=self.hist_by_pattern,
                                    finish=None),
            'HIST_BY_PATTERN_BY_QUALITY': Role(name='HIST_BY_PATTERN_BY_QUALITY', deep=0,
                                               init=self.init_hist_by_pattern_by_quality,
                                               operation=self.hist_by_pattern_by_quality,
                                               finish=None),
            'BAR_BY_PATTERN_BY_QUALITY': Role(name='BAR_BY_PATTERN_BY_QUALITY', deep=0,
                                              init=self.init_bar_by_pattern_by_quality,
                                              operation=self.bar_by_pattern_by_quality,
                                              finish=None),
            'HIST_BY_PATTERN_FULL_FRAME': Role(name='HIST_BY_PATTERN_FULL_FRAME', deep=0,
                                               init=self.init_hist_by_pattern_full_frame,
                                               operation=self.hist_by_pattern_full_frame,
                                               finish=None),
            'BAR_BY_PATTERN_FULL_FRAME': Role(name='BAR_BY_PATTERN_FULL_FRAME', deep=0,
                                              init=self.init_bar_by_pattern_full_frame,
                                              operation=self.bar_by_pattern_full_frame,
                                              finish=None),
        }

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)

        self.workfolder = self.video_context.graphs_folder / role_folder
        self.workfolder_data = self.workfolder / 'data'
        self.workfolder_data.mkdir(parents=True, exist_ok=True)

        self.rc_config()

        for self.bins in bins:
            self.run(**kwargs)

    def rc_config(self):
        import matplotlib as mpl
        self.plot_config = self.config.plot_config[self.role.name]
        rc_param = self.plot_config["rc_param"]

        for group in rc_param:
            kwargs = eval(rc_param[group])
            mpl.rc(group, **kwargs)

    def iterate(self, deep=1):
        count = 0
        for self.video_context.video in self.video_context.videos_list:
            self.name = str(self.video_context.video)
            if deep == 1:
                count += 1
                yield count
                continue
            for self.video_context.tiling in self.video_context.tiling_list:
                self.tiling = str(self.video_context.tiling)
                if deep == 2:
                    count += 1
                    yield count
                    continue
                for self.video_context.quality in self.video_context.quality_list:
                    self.quality = str(self.video_context.quality)
                    if deep == 3:
                        count += 1
                        yield count
                        continue
                    for self.video_context.tile in self.video_context.tiles_list:
                        self.tile = str(self.video_context.tile)
                        if deep == 4:
                            count += 1
                            yield count
                            continue
                        for self.video_context.chunk in self.video_context.chunks_list:
                            self.chunk = str(self.video_context.chunk)
                            if deep == 5:
                                count += 1
                                yield count
                                continue

    def init_hist_by_pattern(self):
        # Load dectime
        self.dectime_filename = self.video_context.dectime_json_file
        self.data_file = self.workfolder_data / 'data.pickle'
        self.color_list = eval(self.plot_config["color_list"])
        self.subplot_pos = eval(self.plot_config["subplot_pos"])

        # Start data bucket
        for tiling in self.config['tiling_list']:
            self.data[tiling]['time'] = []
            self.data[tiling]['time_std'] = []
            self.data[tiling]['rate'] = []

    def hist_by_pattern(self, overwrite):
        def main():
            get_data()
            make_fit()
            calc_stats()
            make_dataframe()
            make_plot()

        def get_data():
            print('\n\n====== Get Data ======')
            if self.data_file.exists() and not overwrite:
                warning(f'  The data file "{self.data_file}" exist. Loading date.')
                self.data = load_pickle(self.data_file)
                return

            total = 0
            dectime = load_json(self.dectime_filename)

            for idx in self.iterate(5):
                if self.quality == '0':
                    continue
                print(f'\r  {total}/{idx:07d} ', end='')
                results = dectime[self.name][self.tiling][self.quality]
                results = results[self.tile][self.chunk]

                if results == {}:
                    warning(f'\nThe result for chain {self.video_context.state} '
                            f'is empty. Skipping.')
                    continue

                bitrate: List[float] = results['bitrate']
                times: List[Tuple[float, ...]] = results['dectimes']

                # self.data[tiling]["time"|"time_std"|"rate"]: list[float|int]
                self.data[self.tiling]['time'].append(np.average(times))
                self.data[self.tiling]['time_std'].append(np.std(times))
                self.data[self.tiling]['rate'].append(bitrate)

            del(dectime)

            save_pickle(self.data, self.data_file)

        def make_fit():
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')
            for tiling in self.config['tiling_list']:
                print(f'  Fitting - tiling {tiling}... ', end='')
                data = self.data[tiling]
                for metric in ['time', 'rate']:
                    print(f'{metric}... ', end='')
                    name = f'fitter_{metric}_{tiling}_{self.bins}bins.pickle'
                    fitter_pickle_file = self.workfolder / 'data' / name
                    bins=self.bins
                    if self.bins == 'custom':
                        min_ = np.min(data[metric])
                        max_ = np.max(data[metric])
                        norm = round((max_ - min_) / 0.001)
                        if norm > 60:
                            bins = 60
                        else:
                            bins = norm
                        fitter_pickle_file = fitter_pickle_file.with_stem(fitter_pickle_file.stem + f'_{bins}bins')

                    if fitter_pickle_file.exists() and not overwrite:
                        print(f'Pickle found! ', end='')
                        fitter = load_pickle(fitter_pickle_file)
                    else:
                        distributions = self.config['distributions']
                        fitter = Fitter(data[metric], bins=bins,
                                        distributions=distributions,
                                        timeout=300)
                        fitter.fit()
                        save_pickle(fitter, fitter_pickle_file)

                    self.fitter[tiling][metric] = fitter
                print(f'  Finished.')

        def calc_stats():
            print('  Calculating Statistics')

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

            stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
            if stats_file.exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                self.stats = pd.read_csv(stats_file)
                return

            stats = defaultdict(list)
            for tiling in self.config['tiling_list']:
                # data[tiling]["time"|"time_std"|"rate"]
                data = self.data[tiling]
                corr = np.corrcoef((data['time'], data['rate']))[1][0]

                stats[f'tiling'].append(tiling)
                stats[f'correlation'].append(corr)

                for metric in ['time', 'rate']:
                    # 1. Stats
                    percentile = np.percentile(data[metric],
                                               [0, 25, 50, 75, 100]).T
                    stats[f'{metric}_average'].append(np.average(data[metric]))
                    stats[f'{metric}_std'].append(float(np.std(data[metric])))
                    stats[f'{metric}_min'].append(percentile[0])
                    stats[f'{metric}_quartile1'].append(percentile[1])
                    stats[f'{metric}_median'].append(percentile[2])
                    stats[f'{metric}_quartile3'].append(percentile[3])
                    stats[f'{metric}_max'].append(percentile[4])

                    # 2. Errors and dists
                    # rmse: DataFrame[distribution_name, ...] -> float
                    fitter = self.fitter[tiling][metric]
                    df_errors: pd.DataFrame = fitter.df_errors
                    bins = len(fitter.x)
                    sse:pd.Series = df_errors['sumsquare_error']
                    rmse = np.sqrt(sse / bins)
                    nrmse = rmse / (sse.max() - sse.min())

                    for dist in sse.keys():
                        params = fitter.fitted_param[dist]
                        dist_info = find_dist(dist, params)
                        stats[f'{metric}_rmse_{dist}'].append(rmse[dist])
                        stats[f'{metric}_nrmse_{dist}'].append(nrmse[dist])
                        stats[f'{metric}_sse_{dist}'].append(sse[dist])
                        stats[f'{metric}_param_{dist}'].append(dist_info['parameters'])
                        stats[f'{metric}_loc_{dist}'].append(dist_info['loc'])
                        stats[f'{metric}_scale_{dist}'].append(dist_info['scale'])

            pd.DataFrame(stats).to_csv(str(stats_file), index=False)

        def make_dataframe():
            print('\n====== Make Dataframe ======')

            def main():
                make_df_data()

            def make_df_data():
                print('  Making df_data')
                for tiling in self.config['tiling_list']:
                    path = self.workfolder / f'df_data{tiling}_{self.bins}bins.csv'
                    if path.exists() and not overwrite:
                        warning('Dataframe exist. Skipping.')
                        continue
                    df_data = {f'time_{tiling}': self.data[tiling]['time'],
                               f'rate_{tiling}': self.data[tiling]['rate']}
                    pd.DataFrame(df_data).to_csv(str(path), index=False)

            main()

        def make_plot():
            print(f'\n====== Make Plot - Bins = {self.bins} ======')
            n_dist = 3
            label = 'empirical'
            name = f'hist_by_pattern_{self.bins}bins.png'
            path = self.workfolder / name
            if path.exists() and not overwrite:
                warning(f'Figure exist. Skipping')
                return

            fig = figure.Figure()
            subplot_pos = eval(self.plot_config['subplot_pos'])

            for idx, tiling in enumerate(self.config['tiling_list'], 1):
                index = idx
                nrows, ncols, index = subplot_pos[index-1]
                fitter = self.fitter[tiling]['time']
                fit_errors = fitter.df_errors
                dists = fit_errors['sumsquare_error'].sort_values()[0:n_dist].index

                ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                ax.hist(self.data[tiling]['time'],
                        bins=len(fitter.x),
                        histtype='bar',
                        density=True,
                        label=label,
                        color='#3297c9')

                for n, dist_name in enumerate(dists):
                    fitted_pdf = fitter.fitted_pdf[dist_name]
                    ax.plot(fitter.x, fitted_pdf,
                            label=f'{dist_name}',
                            color=self.color_list[n],
                            lw=1., )
                ax.set_title(f'{tiling}')
                ax.set_xlabel('Decoding Time (s)')
                ax.legend(loc='upper right')
            # fig.show()
            print(f'Saving the figure')
            fig.savefig(path)

        main()

    def end_hist_by_pattern(self):
        pass

    def init_hist_by_pattern_by_quality(self):
        self.dectime_filename = self.video_context.dectime_json_file
        self.data_file = self.workfolder_data / 'data.pickle'
        self.color_list = eval(self.plot_config["color_list"])
        self.subplot_pos = eval(self.plot_config["subplot_pos"])

    def hist_by_pattern_by_quality(self, overwrite):
        def _main():
            get_data()
            make_fit()
            calc_stats()
            make_dataframe()
            make_plot()

        def get_data():
            print('\n\n====== Get Data ======')
            if self.data_file.exists() and not overwrite:
                warning(f'\n  The data file "{self.data_file}" exist. Loading date.')
                self.data = load_pickle(self.data_file)
                return

            total = 0
            dectime = load_json(self.dectime_filename)
            bitrate, times = [],[]

            for idx in self.iterate(5):
                if self.quality == '0':
                    continue
                print(f'\r  {total}/{idx:07d} ', end='')
                results = dectime[self.name][self.tiling][self.quality]
                results = results[self.tile][self.chunk]

                if results == {}:
                    warning(f'\n  The result for chain {self.video_context.state} '
                            f'is empty. Skipping.')
                    continue

                bitrate: List[float] = results['bitrate']
                times: List[List[float]] = results['dectimes']

                # self.data[tiling][quality]["time"|"time_std"|"rate"]: list[float|int]
                if self.data[self.tiling][self.quality] == {}:
                    self.data[self.tiling][self.quality] = defaultdict(list)
                self.data[self.tiling][self.quality]['time'].append(np.average(times))
                self.data[self.tiling][self.quality]['time_std'].append(np.std(times))
                self.data[self.tiling][self.quality]['rate'].append(bitrate)

            del(dectime)

            save_pickle(self.data, self.data_file)

        def make_fit():
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')
            for tiling in self.config['tiling_list']:
                for quality in self.config['quality_list']:
                    if quality == '0': continue

                    print(f'  Fitting - tiling {tiling}/CRF{quality}... ', end='')
                    data = self.data[tiling][quality]
                    for metric in ['time', 'rate']:
                        name = f'fitter_{metric}_{tiling}_crf{quality}_{self.bins}bins.pickle'
                        fitter_pickle_file = self.workfolder_data / name
                        bins=self.bins
                        if self.bins == 'custom':
                            min_ = np.min(data[metric])
                            max_ = np.max(data[metric])
                            norm = round((max_ - min_) / 0.001)
                            if norm > 60:
                                bins = 60
                            else:
                                bins = norm
                            fitter_pickle_file = fitter_pickle_file.with_stem(fitter_pickle_file.stem + f'_{bins}bins')

                        if fitter_pickle_file.exists() and not overwrite:
                            print(f'{metric} pickle found! ', end='')
                            fitter = load_pickle(fitter_pickle_file)
                        else:
                            distributions = self.config['distributions']
                            fitter = Fitter(data[metric], bins=bins,
                                            distributions=distributions,
                                            timeout=300)
                            fitter.fit()
                            save_pickle(fitter, fitter_pickle_file)

                        self.fitter[tiling][quality][metric] = fitter
                    print(f'  Finished.')

        def calc_stats():
            print('  Calculating Statistics')

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

            stats_file = self.workfolder / f'stats_{self.bins}bins.csv'
            if stats_file.exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                self.stats = pd.read_csv(stats_file)
                return

            stats = defaultdict(list)
            for tiling in self.config['tiling_list']:
                for quality in self.config['quality_list']:
                    if quality == '0': continue

                    # data[tiling][quality]["time"|"time_std"|"rate"]
                    data = self.data[tiling][quality]
                    fitter = self.fitter[tiling][quality]
                    corr = np.corrcoef((data['time'], data['rate']))[1][0]

                    stats[f'tiling'].append(tiling)
                    stats[f'quality'].append(quality)
                    stats[f'correlation'].append(corr)

                    for metric in ['time', 'rate']:
                        # 1. Stats
                        percentile = np.percentile(data[metric],
                                                   [0, 25, 50, 75, 100]).T
                        stats[f'{metric}_average'].append(np.average(data[metric]))
                        stats[f'{metric}_std'].append(float(np.std(data[metric])))
                        stats[f'{metric}_min'].append(percentile[0])
                        stats[f'{metric}_quartile1'].append(percentile[1])
                        stats[f'{metric}_median'].append(percentile[2])
                        stats[f'{metric}_quartile3'].append(percentile[3])
                        stats[f'{metric}_max'].append(percentile[4])

                        # 2. Errors and dists
                        # rmse: DataFrame[distribution_name, ...] -> float
                        df_errors: pd.DataFrame = fitter[metric].df_errors
                        bins = len(fitter[metric].x)
                        sse:pd.Series = df_errors['sumsquare_error']
                        rmse = np.sqrt(sse / bins)
                        nrmse = rmse / (sse.max() - sse.min())

                        for dist in sse.keys():
                            params = fitter[metric].fitted_param[dist]
                            dist_info = find_dist(dist, params)
                            stats[f'{metric}_rmse_{dist}'].append(rmse[dist])
                            stats[f'{metric}_nrmse_{dist}'].append(nrmse[dist])
                            stats[f'{metric}_sse_{dist}'].append(sse[dist])
                            stats[f'{metric}_param_{dist}'].append(dist_info['parameters'])
                            stats[f'{metric}_loc_{dist}'].append(dist_info['loc'])
                            stats[f'{metric}_scale_{dist}'].append(dist_info['scale'])

            pd.DataFrame(stats).to_csv(str(stats_file), index=False)

        def make_dataframe():
            print('\n====== Make Dataframe ======')

            def make_df_data():
                print('  Making df_data')
                for tiling in self.config['tiling_list']:
                    for quality in self.config['quality_list']:
                        if quality == '0': continue
                        path = self.workfolder / f'df_data{tiling}_CRF{quality}.csv'
                        if path.exists() and not overwrite: continue
                        df_data = {f'time': self.data[tiling][quality]['time'],
                                   f'rate': self.data[tiling][quality]['rate']}
                        pd.DataFrame(df_data).to_csv(str(path), index=False)

            make_df_data()

        def make_plot():
            print('\n====== Make Plot - Bins = {self.bins} ======')
            n_dist = 3
            label = 'empirical'
            subplot_pos = eval(self.plot_config['subplot_pos'])

            for quality in self.config['quality_list']:
                if quality == '0': continue

                name = f'hist_by_pattern_CRF{quality}_{self.bins}bins.png'
                path = self.workfolder / name
                if path.exists() and not overwrite: return

                fig = figure.Figure()

                for idx, tiling in enumerate(self.config['tiling_list'], 1):
                    nrows, ncols, index = subplot_pos[idx-1]
                    fitter = self.fitter[tiling][quality]['time']
                    fit_errors = fitter.df_errors
                    dists = fit_errors['sumsquare_error'].sort_values()[0:n_dist].index

                    ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                    ax.hist(self.data[tiling][quality]['time'],
                            bins=len(fitter.x),
                            histtype='bar',
                            density=True,
                            label=label,
                            color='#3297c9')

                    for n, dist_name in enumerate(dists):
                        fitted_pdf = fitter.fitted_pdf[dist_name]
                        ax.plot(fitter.x, fitted_pdf,
                                label=f'{dist_name}',
                                color=self.color_list[n],
                                lw=1., )
                    ax.set_title(f'{tiling} - CRF {quality}')
                    ax.set_xlabel('Decoding Time (s)')
                    ax.legend(loc='upper right')
                # fig.show()
                print(f'  Saving the figure')
                fig.savefig(path)

        _main()

    def init_bar_by_pattern_by_quality(self):
        self.dectime_filename = self.video_context.dectime_json_file
        self.color_list = eval(self.plot_config["color_list"])
        self.subplot_pos = eval(self.plot_config["subplot_pos"])

        self.hist_workfolder = self.video_context.graphs_folder / 'HistByPatternByQuality'
        self.hist_workfolder_data = self.hist_workfolder / 'data'
        self.data_file = self.hist_workfolder_data / 'data.pickle'

    def bar_by_pattern_by_quality(self, overwrite):
        def _main():
            get_data()
            make_fit()
            calc_stats()
            make_plot()

        def get_data():
            print('\n\n====== Get Data ======')
            if self.data_file.exists() and not overwrite:
                warning(f'\n  The data file "{self.data_file}" exist. Loading date.')
                self.data = load_pickle(self.data_file)
                return

            total = 0
            dectime = load_json(self.dectime_filename)
            bitrate, times = [],[]

            for idx in self.iterate(5):
                if self.quality == '0':
                    continue
                print(f'\r  {total}/{idx:07d} ', end='')
                results = dectime[self.name][self.tiling][self.quality]
                results = results[self.tile][self.chunk]

                if results == {}:
                    warning(f'\n  The result for chain {self.video_context.state} '
                            f'is empty. Skipping.')
                    continue

                bitrate: List[float] = results['bitrate']
                times: List[List[float]] = results['dectimes']

                # self.data[tiling][quality]["time"|"time_std"|"rate"]: list[float|int]
                if self.data[self.tiling][self.quality] == {}:
                    self.data[self.tiling][self.quality] = defaultdict(list)
                self.data[self.tiling][self.quality]['time'].append(np.average(times))
                self.data[self.tiling][self.quality]['time_std'].append(np.std(times))
                self.data[self.tiling][self.quality]['rate'].append(bitrate)

            del(dectime)

            save_pickle(self.data, self.data_file)

        def make_fit():
            print(f'\n\n====== Make Fit - Bins = {self.bins} ======')
            for tiling in self.config['tiling_list']:
                for quality in self.config['quality_list']:
                    if quality == '0': continue

                    print(f'  Fitting - tiling {tiling}/CRF{quality}... ', end='')
                    data = self.data[tiling][quality]
                    for metric in ['time', 'rate']:
                        name = f'fitter_{metric}_{tiling}_crf{quality}_{self.bins}bins.pickle'
                        fitter_pickle_file = self.hist_workfolder_data / name
                        bins=self.bins
                        if self.bins == 'custom':
                            min_ = np.min(data[metric])
                            max_ = np.max(data[metric])
                            norm = round((max_ - min_) / 0.001)
                            if norm > 60:
                                bins = 60
                            else:
                                bins = norm
                            fitter_pickle_file = fitter_pickle_file.with_stem(fitter_pickle_file.stem + f'_{bins}bins')

                        if fitter_pickle_file.exists() and not overwrite:
                            print(f'{metric} pickle found! ', end='')
                            fitter = load_pickle(fitter_pickle_file)
                        else:
                            distributions = self.config['distributions']
                            fitter = Fitter(data[metric], bins=bins,
                                            distributions=distributions,
                                            timeout=300)
                            fitter.fit()
                            save_pickle(fitter, fitter_pickle_file)

                        self.fitter[tiling][quality][metric] = fitter
                    print(f'  Finished.')

        def _make_bar(
                plot_type: Union[str, str],
                ax: axes.Axes,
                x: List[float],
                y: List[float],
                yerr: List[float] = None,
                title: str = None,
                label: str = None,
                xlabel: str = None,
                ylabel: str = None,
                xticks: list = None,
                legend: dict = None,
                width: int = 0.8,
                scilimits: Tuple[int] = (6, 6)):

            if plot_type == 'bar':
                ax.bar(x, y, width=width, yerr=yerr, label=label)
            elif plot_type == 'plot':
                ax.plot(x, y, color='r', linewidth=1, label=label)

            ax.ticklabel_format(axis='y', style='scientific',
                                scilimits=scilimits)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if xticks is not None:
                ax.set_xticks(x)
                ax.set_xticklabels(xticks)
            if title is not None:
                ax.set_title(title)
            if legend is not None:
                ax.legend(**legend)

        def calc_stats():
            print('  Calculating Statistics')

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

            stats_file = self.hist_workfolder / f'stats_{self.bins}bins.csv'
            if stats_file.exists() and not overwrite:
                print(f'  stats_file found! Skipping.')
                self.stats = pd.read_csv(stats_file)
                return

            stats = defaultdict(list)
            for tiling in self.config['tiling_list']:
                for quality in self.config['quality_list']:
                    if quality == '0': continue

                    # data[tiling][quality]["time"|"time_std"|"rate"]
                    data = self.data[tiling][quality]
                    fitter = self.fitter[tiling][quality]
                    corr = np.corrcoef((data['time'], data['rate']))[1][0]

                    stats[f'tiling'].append(tiling)
                    stats[f'quality'].append(quality)
                    stats[f'correlation'].append(corr)

                    for metric in ['time', 'rate']:
                        # 1. Stats
                        percentile = np.percentile(data[metric],
                                                   [0, 25, 50, 75, 100]).T
                        stats[f'{metric}_average'].append(np.average(data[metric]))
                        stats[f'{metric}_std'].append(float(np.std(data[metric])))
                        stats[f'{metric}_min'].append(percentile[0])
                        stats[f'{metric}_quartile1'].append(percentile[1])
                        stats[f'{metric}_median'].append(percentile[2])
                        stats[f'{metric}_quartile3'].append(percentile[3])
                        stats[f'{metric}_max'].append(percentile[4])

                        # 2. Errors and dists
                        # rmse: DataFrame[distribution_name, ...] -> float
                        df_errors: pd.DataFrame = fitter[metric].df_errors
                        bins = len(fitter[metric].x)
                        sse:pd.Series = df_errors['sumsquare_error']
                        rmse = np.sqrt(sse / bins)
                        nrmse = rmse / (sse.max() - sse.min())

                        for dist in sse.keys():
                            params = fitter[metric].fitted_param[dist]
                            dist_info = find_dist(dist, params)
                            stats[f'{metric}_rmse_{dist}'].append(rmse[dist])
                            stats[f'{metric}_nrmse_{dist}'].append(nrmse[dist])
                            stats[f'{metric}_sse_{dist}'].append(sse[dist])
                            stats[f'{metric}_param_{dist}'].append(dist_info['parameters'])
                            stats[f'{metric}_loc_{dist}'].append(dist_info['loc'])
                            stats[f'{metric}_scale_{dist}'].append(dist_info['scale'])

            pd.DataFrame(stats).to_csv(str(stats_file), index=False)

        def make_plot():
            subplot_pos = eval(self.plot_config['subplot_pos'])

            name = f'Bar_by_pattern_by_quality_{self.bins}bins.png'
            path = self.workfolder / name
            if path.exists() and not overwrite:
                return
            fig = figure.Figure()
            
            for idx, tiling in enumerate(self.config['tiling_list'], 1):
                nrows, ncols, index = subplot_pos[idx-1]
                ax: axes.Axes = fig.add_subplot(nrows, ncols, index)
                data = self.stats[self.stats['tiling'] == tiling]
                # bar plot of lib
                x = xticks = data['quality']
                time_avg = data['time_average']
                time_std = data['time_std']
                ylabel = 'Decoding time (s)' if idx in [1, 2, 4] else None
                _make_bar('bar', ax=ax,
                            x=x, y=time_avg, yerr=time_std,
                            title=f'{tiling}',
                            xlabel='CRF',
                            ylabel=ylabel,
                            xticks=xticks,
                            width=5,
                            scilimits=None)

                # line plot of bit rate
                ax = ax.twinx()
                rate_avg = data['rate_average']
                patch = mpatches.Patch(color='#1f77b4', label='Time')
                line = mlines.Line2D([], [], color='red', label='Bitrate')
                legend = {'handles': (patch, line),
                          'loc'    : 'upper right'}
                ylabel = 'Bit Rate (Mbps)' if idx in [1, 5] else None

                _make_bar('plot', ax=ax,
                            x=x, y=rate_avg,
                            legend=legend,
                            ylabel=ylabel,
                          scilimits=(6, 6))
            # fig.show()
            print(f'Salvando a figura')
            fig.savefig(path)

        _main()

    def init_hist_by_pattern_full_frame(self): ...
    def hist_by_pattern_full_frame(self): ...

    def init_bar_by_pattern_full_frame(self): ...
    def bar_by_pattern_full_frame(self): ...


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
        }

        self.role = operations[role]
        self.check_table = {'file':[],'msg':[]}

        self.results = AutoDict()
        self.results_dataframe = pd.DataFrame()
        self.config = Config(config)
        self.video_context = VideoContext(self.config, self.role.deep)

        try:
            self.run(**kwargs)
        except KeyboardInterrupt:
            self.save()
            raise KeyboardInterrupt

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
        for factor in self.video_context.state:
            results = results[factor]

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
        if self.video_context.video.projection == 'equirectangular':
            height, width = self.video_context.video.resolution.shape
            func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi
                                       / height)
            self.weight_ndarray = np.fromfunction(func, (height, width),
                                                  dtype='float32')
        elif self.video_context.video.projection == 'cubemap':
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
            info('Skipping original quality')
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

    def init_result(self):
        quality_result_json = self.video_context.quality_result_json

        if quality_result_json.exists():
            warning(f'The file {quality_result_json} exist. Loading.')
            json_content = quality_result_json.read_text(encoding='utf-8')
            self.results = load_json(json_content)
        else:
            self.results = AutoDict()

    def result(self, overwrite=False):
        debug(f'Processing {self.video_context}')
        if self.video_context.quality == self.video_context.original_quality:
            info('Skipping original quality')
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
        return 'continue'

    def save_result(self):
        quality_result_json = self.video_context.quality_result_json
        save_json(self.results, quality_result_json)

        pickle_dumps = pickle.dumps(self.results)
        quality_result_pickle = quality_result_json.with_suffix('.pickle')
        quality_result_pickle.write_bytes(pickle_dumps)

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


class GetTiles(BaseTileDecodeBenchmark):
    results = AutoDict()
    database = AutoDict()
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
            'GET_TILES': Role(name='GET_TILES', deep=2,
                              init=self.init_get_tiles,
                              operation=self.get_tiles,
                              finish=self.finish_get_tiles),
            'JSON2PICKLE': Role(name='JSON2PICKLE', deep=2,
                                init=None,
                                operation=self.json2pickle,
                                finish=None),
            'REFACTOR': Role(name='REFACTOR', deep=2, init=self.init_refactor,
                             operation=self.refactor,
                             finish=self.end_refactor),
        }
        ds_name = kwargs['dataset_name']

        self.config = Config(config)
        self.role = operations[role]
        self.video_context = VideoContext(self.config, self.role.deep)

        self.dataset_name = ds_name
        self.database_folder = Path('datasets')
        self.database_path = self.database_folder / ds_name
        self.database_json = self.database_path / f'{ds_name}.json'
        self.database_pickle = self.database_path / f'{ds_name}.pickle'

        self.get_tiles_path = (self.video_context.project_path
                               / self.video_context.get_tiles_folder)
        self.get_tiles_path.mkdir(parents=True, exist_ok=True)

        self.run(**kwargs)

    # Refactor
    def init_refactor(self):
        self.video_context.dataset_name = self.dataset_name
        self.hm_dataset = load_pickle(self.video_context.dataset_pickle)
        self.dectime = load_json(self.video_context.dectime_json_file)
        self.tile_quality = load_json(self.video_context.quality_result_json)

    def refactor(self, **kwargs):
        print(f'{self.video_context.state}')
        if self.video_context.video != self.old_video:
            if self.old_video is not None:
                new_database_pickle = (self.video_context.project_path
                                       / f'database_{self.name}.pickle')
                save_pickle(self.new_database, new_database_pickle)

            self.old_video = self.video_context.video
            self.old_name = f'{self.video_context.video}'

            name = self.old_name.replace('_cmp', '')
            self.name = name.replace('_erp', '')
            self.projection = self.video_context.video.projection

            new_database_pickle = (self.video_context.project_path
                                   / f'database_{self.name}.pickle')

            if new_database_pickle.exists():
                self.new_database = load_pickle(new_database_pickle)
            else:
                self.new_database = AutoDict()

            self.new_database[self.projection]['dectime_rate'] |= self.dectime[self.old_name]
            self.new_database[self.projection]['tile_quality'] |= self.tile_quality[self.name]
            self.new_database['hm_dataset'][self.dataset_name] |= self.hm_dataset[self.name]

        ##########################
        if self.video_context.tiling != self.old_tiling:
            self.old_tiling = self.video_context.tiling
            self.tiling = f'{self.video_context.tiling}'

            if self.tiling != '1x1':
                get_tiles = load_pickle(self.video_context.get_tiles_pickle)
                for user_id in get_tiles:
                    self.new_database[self.projection]['get_tiles'][
                        self.tiling][user_id] = get_tiles[user_id][:1800]
            else:
                for user_id in self.hm_dataset[self.name]:
                    self.new_database[self.projection]['get_tiles'][
                        self.tiling][user_id] = [[0]]*1800

    def end_refactor(self):
        new_database_pickle = (self.video_context.project_path
                               / f'database_{self.name}.pickle')
        save_pickle(self.new_database, new_database_pickle)

    def process_nasrabadi(self, **kwargs):
        overwrite = kwargs['overwrite']
        database_pickle = self.database_pickle

        if database_pickle.exists() and not overwrite:
            warning(f'The file {database_pickle} exist. Skipping.')
            return

        database = AutoDict()

        video_id_map_csv = pd.read_csv(f'{self.database_path}/video_id_map.csv')
        video_id_map = video_id_map_csv['my_id']
        video_id_map.index = video_id_map_csv['video_id']

        # "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
        # respectively, to reorient during playback." - Author
        nasrabadi_rotation = {10: np.deg2rad(-265), 17: np.deg2rad(-180),
                              27: np.deg2rad(-63), 28: np.deg2rad(-81)}

        user_map = {}

        for csv_database_file in self.database_path.glob('*/*.csv'):
            info('Preparing Local.')
            user, video_id = csv_database_file.stem.split('_')
            video_name = video_id_map[video_id]

            info(f'Renaming users.')
            if user not in user_map.keys():
                user_map[user] = len(user_map)
            user_id = str(user_map[user])

            info(f'Checking if database key was collected.')
            if database[video_name][user_id] != {} and not overwrite:
                warning(f'The key [{video_name}][{user_id}] is not empty. '
                        f'Skipping')
                continue

            info(f'Loading original database.')
            head_movement = pd.read_csv(csv_database_file,
                                        names=['timestamp',
                                               'Qx', 'Qy', 'Qz', 'Qw',
                                               'Vx', 'Vy', 'Vz'])

            rotation = 0
            if video_id in [10,17,27,28]:
                info(f'Cheking rotation offset.')
                rotation = nasrabadi_rotation[video_id]

            frame_counter = 0
            last_line = None
            theta, phi = [], []
            start_time = time.time()

            for n, line in enumerate(head_movement.iterrows()):
                if not len(theta) < 1800: break
                line = line[1]
                print(f'\rUser {user_id} - {video_name} - sample {n:04d} - frame {frame_counter}', end='')

                # Se o timestamp for menor do que frame_time,
                # continue. Senão, faça interpolação. frame time = counter / fps
                frame_time = frame_counter / 30
                if line.timestamp < frame_time:
                    last_line = line
                    continue
                if line.timestamp == frame_time:
                    # Based on gitHub code of author
                    x, y, z = line[['Vz', 'Vx', 'Vy']]
                else:
                    # Linear Interpolation
                    t: float = frame_time
                    t_f: float = line.timestamp
                    t_i: float = last_line.timestamp
                    v_f: pd.Serie = line[['Vz', 'Vx', 'Vy']]
                    v_i: pd.Serie = last_line[['Vz', 'Vx', 'Vy']]
                    x, y, z = lin_interpol(t, t_f, t_i, v_f, v_i)

                azimuth, elevation = xyz2hcs(x, y, z)
                new_azimuth = azimuth + rotation
                if new_azimuth < -np.pi:
                    new_azimuth += 2 * np.pi
                elif new_azimuth >= np.pi:
                    new_azimuth -= 2 * np.pi

                if elevation < -np.pi / 2:
                    elevation = - np.pi - elevation
                elif elevation > np.pi / 2:
                    elevation = np.pi - elevation

                theta.append(new_azimuth)
                phi.append(elevation)
                frame_counter += 1
                last_line = line

            database[video_name][user_id] = {'azimuth': theta, 'elevation': phi}
            print(f' - {time.time() - start_time:0.3f}')

        print(f'\nFinish. Saving as {database_pickle}.')
        save_pickle(database, database_pickle)

    def init_get_tiles(self):
        for video in self.config.videos_list:
            if self.config.videos_list[video]['projection'] == 'equirectangular':
                # self.config.videos_list[video]['scale'] = '144x72'
                # self.config.videos_list[video]['scale'] = '288x144'
                # self.config.videos_list[video]['scale'] = '432x216'
                self.config.videos_list[video]['scale'] = '576x288'
            elif self.config.videos_list[video]['projection'] == 'cubemap':
                # self.config.videos_list[video]['scale'] = '144x96'
                # self.config.videos_list[video]['scale'] = '288x192'
                # self.config.videos_list[video]['scale'] = '432x288'
                self.config.videos_list[video]['scale'] = '576x384'

        database_pickle = self.database_pickle
        info(f'Loading database {database_pickle}.')
        self.database = load_pickle(database_pickle)

    def get_tiles(self, **kwargs):
        overwrite = kwargs['overwrite']

        tiling = self.video_context.tiling
        if str(tiling) == '1x1':
            info(f'skipping tiling 1x1')
            return

        video_name = self.video_context.video.name.replace("_cmp", "").replace("_erp", "")
        tiling.fov = '90x90'
        dbname = self.dataset_name
        database = self.database
        self.results = AutoDict()

        filename = f'get_tiles_{dbname}_{video_name}_{self.video_context.video.projection}_{tiling}.pickle'
        get_tiles_pickle = self.get_tiles_path / filename
        info(f'Defined the result filename to {get_tiles_pickle}')

        if get_tiles_pickle.exists() and not overwrite:
            warning(f'The file {get_tiles_pickle} exist. Skipping.')
            return

        print(f'{video_name} - tiling {tiling}')

        for n, user_id in enumerate(database[video_name]):
            # 'timestamp', 'Qx', 'Qy', 'Qz', 'Qw','Vx', 'Vy', 'Vz', 'azimuth',
            # 'elevation'
            yaw = database[video_name][user_id]['azimuth']
            pitch = database[video_name][user_id]['elevation']
            roll = [0]*len(pitch)

            self.results[user_id] = []

            for frame, position in enumerate(zip(yaw, pitch, roll)):
                print(f'\rUser {user_id} - sample {frame:05d}', end='')
                tiles_selected = tiling.get_vptiles(position)
                self.results[user_id].append(tiles_selected)

        print('')
        save_pickle(self.results, get_tiles_pickle)

    def finish_get_tiles(self):
        pass

    def json2pickle(self, overwrite=False):
        video_name = self.video_context.video.name
        tiling = self.video_context.tiling
        dbname = self.dataset_name

        get_tiles_pickle = self.get_tiles_path / f'get_tiles_{dbname}_{video_name}_{self.video_context.video.projection}_{tiling}.pickle'
        get_tiles_json = self.get_tiles_path / f'get_tiles_{dbname}_{video_name}_{self.video_context.video.projection}_{tiling}.json'

        if get_tiles_json.exists() and not get_tiles_pickle.exists() and not overwrite:
            result = load_json(get_tiles_json)
            save_pickle(result, get_tiles_pickle)

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

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)