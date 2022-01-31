import datetime
import json
import pickle
import time
from builtins import PermissionError
from collections import Counter
from logging import warning, debug, fatal, info
from pathlib import Path
from subprocess import run, DEVNULL
from typing import Any, NamedTuple, Union, Dict

import numpy as np
import pandas as pd

from .assets import AutoDict, Role
from .util import (run_command, check_video_gop, iter_frame, load_sph_file,
                   xyz2hcs, save_json, load_json, lin_interpol, save_pickle,
                   load_pickle)
from .video_state import Config, VideoContext


class BaseTileDecodeBenchmark:
    config: Config = None
    state: VideoContext = None
    role: Role = None

    def run(self, **kwargs):
        self.print_resume()
        self.role.init()

        total = len(self.state)
        for n in self.state:
            print(f'{n}/{total}', end='\r', flush=True)
            info(f'\n{self.state.factors_list}')
            action = self.role.operation(**kwargs)

            if action in (None, 'continue', 'skip'):
                continue
            elif action in ('break',):
                break

        self.role.finish()
        print(f'The end of {self.role.name}')

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.role.name}\n'
              f'  project: {self.state.project}\n'
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
        dectime_log = self.state.dectime_log
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

    def get_times(self):
        content = self.state.dectime_log.read_text(encoding='utf-8')
        content_lines = content.splitlines()
        times = [float(line.strip().split(' ')[1].split('=')[1][:-1])
                 for line in content_lines if 'utime' in line]
        return times


class TileDecodeBenchmark(BaseTileDecodeBenchmark):
    results = AutoDict()
    results_dataframe = pd.DataFrame()

    def __init__(self, config: str = None, role: str = None, **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """

        operations = {
            'PREPARE': Role(name='PREPARE', deep=1, init=None,
                            operation=self.prepare, finish=None),
            'COMPRESS': Role(name='COMPRESS', deep=4, init=None,
                             operation=self.compress, finish=None),
            'SEGMENT': Role(name='SEGMENT', deep=4, init=None,
                            operation=self.segment, finish=None),
            'DECODE': Role(name='DECODE', deep=5, init=None,
                           operation=self.decode, finish=None),
            'COLLECT_RESULTS': Role(name='COLLECT_RESULTS', deep=5,
                                    init=self.init_collect_dectime,
                                    operation=self.collect_dectime,
                                    finish=self.save_dectime),
        }

        self.config = Config(config)
        self.role = operations[role]
        self.state = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    # PREPARE
    def prepare(self, overwrite=False) -> Any:
        """
        Prepare video to encode. Uncompress and unify resolution, frame rate,
        pixel format.
        :param overwrite: If True, this method overwrite previous files.
        """
        original = self.state.original_file
        uncompressed_file = self.state.lossless_file
        lossless_log = self.state.lossless_file.with_suffix('.log')

        debug(f'==== Processing {uncompressed_file} ====')

        if uncompressed_file.exists() and not overwrite:
            warning(f'The file {uncompressed_file} exist. Skipping.')
            return 'skip'

        if not original.exists():
            warning(f'The file {original} not exist. Skipping.')
            return 'skip'

        video = self.state
        fps = self.state.fps
        resolution = self.state.resolution
        dar = resolution.W / resolution.H

        cmd = f'ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {video.offset} '
        cmd += f'-i {original} '
        cmd += f'-crf 0 '
        cmd += f'-t {video.duration} '
        cmd += f'-r {fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf "scale={resolution},setdar={dar}" '
        cmd += f'{uncompressed_file}'

        run_command(cmd, lossless_log, 'w')

    # COMPRESS
    def compress(self, overwrite=False) -> Any:
        """
        Encode videos using h.265
        :param overwrite:
        :return:
        """
        uncompressed_file = self.state.lossless_file
        compressed_file = self.state.compressed_file
        compressed_log = self.state.compressed_file.with_suffix('.log')

        debug(f'==== Processing {compressed_file} ====')

        if compressed_file.exists() and not overwrite:
            warning(f'The file {compressed_file} exist. Skipping.')
            return 'skip'

        if not uncompressed_file.exists():
            warning(f'The file {uncompressed_file} not exist. Skipping.')
            return 'skip'

        quality = self.state.quality
        gop = self.state.gop
        tile = self.state.tile

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
        segment_log = self.state.segment_file.with_suffix('.log')
        segment_folder = self.state.segment_folder
        compressed_file = self.state.compressed_file

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
        segment_file = self.state.segment_file
        dectime_log = self.state.dectime_log
        info(f'==== Processing {dectime_log} ====')

        diff = self.state.decoding_num
        if self.state.dectime_log.exists():
            count = self.count_decoding()
            diff = self.state.decoding_num - count
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
    def init_collect_dectime(self) -> Any:
        dectime_json_file = self.state.dectime_json_file

        if dectime_json_file.exists():
            self.results = load_json(dectime_json_file)

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
        debug(f'Collecting {self.state}')

        results = self.results
        for factor in self.state.factors_list:
            results = results[factor]

        if not results == {} and not overwrite:
            warning(f'The result key for {self.state} contain some value. '
                    f'Skipping.')
            return 'skip'  # if value exist and not overwrite, then skip

        if not self.state.segment_file.exists():
            warning(f'The file {self.state.segment_file} not exist. Skipping.')
            return 'skip'

        if not self.state.dectime_log.exists():
            warning(f'The file {self.state.dectime_log} not exist. Skipping.')
            return 'skip'

        try:
            chunk_size = self.state.segment_file.stat().st_size
        except PermissionError:
            warning(f'PermissionError error on reading size of '
                    f'{self.state.segment_file}. Skipping.')
            return 'skip'

        bitrate = chunk_size * 8 / (self.state.gop / self.state.fps)

        times = self.get_times()

        data = {'bitrate': bitrate, 'dectimes': times}
        results.update(data)

    def save_dectime(self):
        filename = self.state.dectime_json_file
        info(f'Saving {filename}')
        save_json(self.results, filename)
        # self.json2pd()

    # def json2pd(self):
    #     """
    #     old function. Maintained for compatibility
    #     name_scale_fps_pattern_"CRF"quality_tile| chunk1 | chunk2 | ... |
    #     average | std | median
    #
    #     :return:
    #     """
    #     results_dataframe = pd.DataFrame(columns=self.state.chunk_list)
    #     for _ in self.iterate(deep=4):
    #         name, pattern, quality, tile, *_ = self.state.get_factors()
    #
    #         results = self.results[name][pattern][quality][tile]
    #         chunks_values = [results[chunk] for chunk in self.state.chunk_list]
    #         results_dataframe.loc[self.state.state] = chunks_values
    #     self.results_dataframe = pd.DataFrame(results_dataframe)


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
        self.check_table = {'file': [], 'msg': []}

        self.results = AutoDict()
        self.results_dataframe = pd.DataFrame()
        self.config = Config(config)
        self.state = VideoContext(self.config, self.role.deep)

        try:
            self.run(**kwargs)
        except KeyboardInterrupt:
            self.save()
            raise KeyboardInterrupt

    def check_original(self, **check_video_kwargs):
        original_file = self.state.original_file
        debug(f'==== Checking {original_file} ====')
        msg = self.check_video(original_file, **check_video_kwargs)

        self.check_table['file'].append(original_file)
        self.check_table['msg'].append(msg)

    def check_lossless(self, **check_video_kwargs):
        lossless_file = self.state.lossless_file
        debug(f'Checking the file {lossless_file}')

        duration = self.state.duration
        fps = self.state.fps
        log_pattern = f'frame={duration * fps:5}'

        msg = self.check_video(lossless_file, log_pattern, **check_video_kwargs)

        self.check_table['file'].append(lossless_file)
        self.check_table['msg'].append(msg)

    def check_compress(self, only_error=True, **check_video_kwargs):
        video_file = self.state.compressed_file
        debug(f'Checking the file {video_file}')

        duration = self.state.duration
        fps = self.state.fps
        log_pattern = f'encoded {duration * fps} frames'

        msg = self.check_video(video_file, log_pattern, **check_video_kwargs)

        if not (only_error and msg == 'log_ok-video_ok'):
            self.check_table['file'].append(video_file)
            self.check_table['msg'].append(msg)

    def check_segment(self, only_error=True, **kwargs):
        segment_file = self.state.segment_file
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
                    if max_gop != self.state.gop:
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
        dectime_log = self.state.dectime_log
        debug(f'Checking the file {dectime_log}')

        if not dectime_log.exists():
            warning('logfile_not_found')
            count_decode = 0
        else:
            count_decode = self.count_decoding()

        if count_decode == 0 and clean:
            dectime_log.unlink(missing_ok=True)

        if not (only_error and count_decode >= self.state.decoding_num):
            msg = f'decoded_{count_decode}x'
            self.check_table['file'].append(dectime_log)
            self.check_table['msg'].append(msg)

    def load_results(self):
        dectime_json_file = self.state.dectime_json_file
        self.results = load_json(dectime_json_file)

    def check_dectime(self, only_error=True):
        results = self.results
        for factor in self.state.factors_list:
            results = results[factor]

        if not (results == {}):
            bitrate = float(results['bitrate'])
            if bitrate > 0:
                msg = 'bitrate_ok'
            else:
                msg = 'bitrate==0'

            dectimes = results['dectimes']
            if len(dectimes) >= self.state.decoding_num:
                msg += '_dectimes_ok'
            else:
                msg += '_dectimes==0'

        else:
            warning(f'The result key for {self.state} is empty.')
            msg = 'empty_key'

        if not (only_error and msg == 'bitrate_ok_dectimes_ok'):
            key = self.state.make_name()
            self.check_table['key'].append(key)
            self.check_table['msg'].append(msg)

    def save(self):
        # Create Paths
        date = datetime.datetime.today()
        table_filename = f'{self.role.name}-table-{date}.csv'
        resume_filename = f'{self.role.name}-resume-{date}.csv'
        check_folder = self.state.check_folder
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
#         self.state = VideoContext(self.config, self.role.deep)
#         self.config['quality_list'] = [28]
#
#         self.run(**kwargs)
#
#     # 'CALCULATE SITI'
#     def siti(self, overwrite=False, animate_graph=False, save=True):
#         if not self.state.compressed_file.exists():
#             if not self.state.lossless_file.exists():
#                 warning(f'The file {self.state.lossless_file} not exist. '
#                         f'Skipping.')
#                 return 'skip'
#             self.compress()
#
#         siti = SiTi(self.state)
#         siti.calc_siti(animate_graph=animate_graph, overwrite=overwrite,
#                        save=save)
#
#     def compress(self):
#         compressed_file = self.state.compressed_file
#         compressed_log = self.state.compressed_file.with_suffix('.log')
#
#         debug(f'==== Processing {compressed_file} ====')
#
#         quality = self.state.quality
#         gop = self.state.gop
#         tile = self.state.tile
#
#         cmd = ['ffmpeg -hide_banner -y -psnr']
#         cmd += [f'-i {self.state.lossless_file}']
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
#         for name in enumerate(self.state.names_list):
#             """Join siti_results"""
#             siti_results_file = self.state.siti_results
#             siti_results_df = pd.read_csv(siti_results_file)
#             if num_frames is None:
#                 num_frames = self.state.duration * self.state.fps
#             elif num_frames < len(siti_results_df['si']):
#                 dif = len(siti_results_df['si']) - num_frames
#                 for _ in range(dif):
#                     siti_results_df.loc[len(siti_results_df)] = [0, 0]
#
#             siti_results_final[f'{name}_ti'] = siti_results_df['si']
#             siti_results_final[f'{name}_si'] = siti_results_df['ti']
#
#             """Join stats"""
#             siti_stats_json_final[name] = load_json(self.state.siti_stats)
#         # siti_results_final.to_csv(f'{self.state.siti_folder /
#         # "siti_results_final.csv"}', index_label='frame')
#         # pd.DataFrame(siti_stats_json_final).to_csv(f'{self.state.siti_folder
#         # / "siti_stats_final.csv"}')
#
#     def _scatter_plot_siti(self):
#         siti_results_df = pd.read_csv(
#             f'{self.state.siti_folder / "siti_stats_final.csv"}', index_col=0)
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
#         fig.savefig(self.state.siti_folder / 'scatter.png')
#         fig.show()


class QualityMetrics:
    PIXEL_MAX: int = 255
    state: VideoContext = None
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
        if self.state.projection == 'equirectangular':
            height, width = self.state.resolution.shape
            func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi
                                       / height)
            self.weight_ndarray = np.fromfunction(func, (height, width),
                                                  dtype='float32')
        elif self.state.projection == 'cubemap':
            # each face must be square (proj. aspect ration == 3:2).
            face = self.state.resolution.shape[0] / 2
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
            fatal(f'projection {self.state.projection} not supported.')
            raise FileNotFoundError(
                f'projection {self.state.projection} not supported.')

    def wspsnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
               im_sal: np.ndarray = None):
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :param im_sal:
        :return:
        """

        if self.state.resolution.shape != self.weight_ndarray.shape:
            self.prepare_weight_ndarray()

        x1 = self.state.tile.position.x
        x2 = self.state.tile.position.x + self.state.tile.resolution.W
        y1 = self.state.tile.position.y
        y2 = self.state.tile.position.y + self.state.tile.resolution.H
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
        shape = self.state.resolution.shape

        if self.sph_points_mask.shape != shape:
            sph_file = load_sph_file(self.sph_file, shape)
            self.sph_points_mask = sph_file[-1]

        x1 = self.state.tile.position.x
        x2 = self.state.tile.position.x + self.state.tile.resolution.W
        y1 = self.state.tile.position.y
        y2 = self.state.tile.position.y + self.state.tile.resolution.H
        mask = self.sph_points_mask[y1:y2, x1:x2]

        im_ref_m = im_ref * mask
        im_deg_m = im_deg * mask

        sqr_dif: np.ndarray = (im_ref_m - im_deg_m) ** 2

        if im_sal is not None:
            sqr_dif = sqr_dif * im_sal

        mse = sqr_dif.sum() / mask.sum()
        return self.mse2psnr(mse)

    def ffmpeg_psnr(self):
        if self.state.chunk == 1:
            name, pattern, quality, tile, chunk = self.state.factors_list
            results = self.results[name][pattern][quality][tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        psnr = None
        compressed_log = self.state.compressed_file.with_suffix('.log')
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
            'ALL': Role(name='ALL', deep=4, init=self.all_init,
                        operation=self.all, finish=None),
            'PSNR': Role(name='PSNR', deep=4, init=self.all_init,
                         operation=self.only_a_metric, finish=None),
            'WSPSNR': Role(name='WSPSNR', deep=4, init=self.all_init,
                           operation=self.only_a_metric, finish=None),
            'SPSNR': Role(name='SPSNR', deep=4, init=self.all_init,
                          operation=self.only_a_metric, finish=None),
            'RESULTS': Role(name='RESULTS', deep=4, init=self.init_result,
                            operation=self.result, finish=self.save_result),
        }

        self.metrics = {'PSNR': self.psnr,
                        'WS-PSNR': self.wspsnr,
                        'S-PSNR': self.spsnr_nn}

        self.sph_file = sph_file

        self.results_dataframe = pd.DataFrame()

        self.config = Config(config)
        self.role = operations[role]
        self.state = VideoContext(self.config, self.role.deep)

        self.run(**kwargs)

    def all_init(self):
        pass
        # quality_pickle = self.state.quality_pickle
        # if quality_pickle.exists():
        #     self.results = pickle.load(quality_pickle.open('rb'))
        # else:
        #     self.results = AutoDict()

    def all(self, overwrite=False):
        debug(f'Processing {self.state}')
        if self.state.quality == self.state.original_quality:
            info('Skipping original quality')
            return 'continue'

        reference_file = self.state.reference_file
        compressed_file = self.state.compressed_file
        quality_csv = self.state.quality_csv

        metrics = self.metrics.copy()

        if quality_csv.exists() and not overwrite:
            csv_dataframe = pd.read_csv(quality_csv, encoding='utf-8',
                                        index_col=0)
            for metric in csv_dataframe:
                info(
                    f'The metric {metric} exist for {self.state.factors_list}. '
                    'Skipping this metric')
                del (metrics[metric])
            if metrics == {}:
                warning(f'The metrics for {self.state.factors_list} are '
                        f'OK. Skipping')
                return
        else:
            csv_dataframe = pd.DataFrame()

        results = {metric: [] for metric in metrics}

        frames = zip(iter_frame(reference_file), iter_frame(compressed_file))
        start = time.time()
        for n, (frame_video1, frame_video2) in enumerate(frames, 1):
            for metric in metrics:
                metrics_method = self.metrics[metric]
                metric_value = metrics_method(frame_video1, frame_video2)
                results[metric].append(metric_value)
            print(
                f'{self.state.factors_list} - Frame {n} - {time.time() - start: 0.3f} s',
                end='\r')
        print()
        for metric in results:
            csv_dataframe[metric] = results[metric]
        csv_dataframe.to_csv(quality_csv, encoding='utf-8', index_label='frame')

    def only_a_metric(self, **kwargs):
        self.metrics = self.metrics[self.role.name]
        self.all(**kwargs)

    def init_result(self):
        quality_result_json = self.state.quality_result_json

        if quality_result_json.exists():
            warning(f'The file {quality_result_json} exist. Loading.')
            json_content = quality_result_json.read_text(encoding='utf-8')
            self.results = load_json(json_content)
        else:
            self.results = AutoDict()

    def result(self, overwrite=False):
        debug(f'Processing {self.state}')
        if self.state.quality == self.state.original_quality:
            info('Skipping original quality')
            return 'continue'

        results = self.results
        quality_csv = self.state.quality_csv  # The compressed quality

        if not quality_csv.exists():
            warning(f'The file {quality_csv} not exist. Skipping.')
            return 'continue'

        csv_dataframe = pd.read_csv(quality_csv, encoding='utf-8', index_col=0)

        for key in self.state.factors_list:
            results = results[key]

        for metric in self.metrics:
            if results[metric] != {} and not overwrite:
                warning(f'The metric {metric} exist for Result '
                        f'{self.state.factors_list}. Skipping this metric')
                return

            try:
                results[metric] = csv_dataframe[metric].tolist()
                if len(results[metric]) == 0:
                    raise KeyError
            except KeyError:
                warning(f'The metric {metric} not exist for csv_dataframe'
                        f'{self.state.factors_list}. Skipping this metric')
                return
        return 'continue'

    def save_result(self):
        quality_result_json = self.state.quality_result_json
        save_json(self.results, quality_result_json)

        pickle_dumps = pickle.dumps(self.results)
        quality_result_pickle = quality_result_json.with_suffix('.pickle')
        quality_result_pickle.write_bytes(pickle_dumps)

    def ffmpeg_psnr(self):
        if self.state.chunk == 1:
            name, pattern, quality, tile, chunk = self.state.factors_list
            results = self.results[name][pattern][quality][tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        psnr = None
        compressed_log = self.state.compressed_file.with_suffix('.log')
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

    def __init__(self, config: str, role: str,
                 **kwargs):
        """
        Load configuration and run the main routine defined on Role Operation.
        todo: Converter o dataset para um formato mais fácil de usar
        todo: pegar os tiles para cada vídeo e para cada segmentação (deep==2)

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

        }
        db_name = kwargs['database_name']

        self.config = Config(config)
        self.role = operations[role]
        self.state = VideoContext(self.config, self.role.deep)

        self.database_name = db_name
        self.database_folder = Path('datasets')
        self.database_path = self.database_folder / db_name
        self.database_json = self.database_path / f'{db_name}.json'
        self.database_pickle = self.database_path / f'{db_name}.pickle'

        self.get_tiles_path = self.state.project / self.state.get_tiles_folder
        self.get_tiles_path.mkdir(parents=True, exist_ok=True)

        self.run(**kwargs)

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
        # respectively, to reorient during playback." - Autor
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
                line = line[1]
                print(f'\rUser {user_id} - {video_name} - sample {n:04d} - frame {frame_counter}', end='')

                # Se o timestamp for menor do que frame_time,
                # continue. Senão, faça interpolação. frame time = counter / fps
                frame_time = frame_counter / 30
                if line.timestamp < frame_time:
                    last_line = line
                    continue

                if line.timestamp == frame_time:
                    # Based on github code of author
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

        tiling = self.state.tiling
        if str(tiling) == '1x1':
            info(f'skipping tiling 1x1')
            return

        video_name = self.state.name
        tiling.fov = '90x90'
        dbname = self.database_name
        database = self.database
        self.results = AutoDict()

        filename = f'get_tiles_{dbname}_{video_name}_{self.state.projection}_{tiling}.pickle'
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

    def json2pickle(self, **kwargs):
        video_name = self.state.name
        tiling = self.state.tiling
        dbname = self.database_name

        get_tiles_pickle = self.get_tiles_path / f'get_tiles_{dbname}_{video_name}_{self.state.projection}_{tiling}.pickle'
        get_tiles_json = self.get_tiles_path / f'get_tiles_{dbname}_{video_name}_{self.state.projection}_{tiling}.json'

        if get_tiles_json.exists() and not get_tiles_pickle.exists():
            result = load_json(get_tiles_json)
            save_pickle(result, get_tiles_pickle)
