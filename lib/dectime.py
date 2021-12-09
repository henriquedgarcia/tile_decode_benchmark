import json
import os
import pickle
from abc import abstractmethod, ABC
from builtins import PermissionError
from collections import Counter, defaultdict
from logging import warning, info, debug
from pathlib import Path
from subprocess import run, DEVNULL
from typing import Union, Any, Dict, List, NamedTuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.siti import SiTi
from lib.util import AutoDict, run_command
import skvideo.io
import datetime


from lib.video_state import Config, VideoContext, Role, Operation


# class Video:
#     def __init__(self, config: Config):
#         # Global
        # if True:
            # self.config = config
            # self.project = config.project
            # self.fps = config.fps
            # self.gop = config.gop
            # self.rate_control = config.rate_control

        # Video dependent
        # if True:
        #     self.original: Path = None
        #     self.frame: Frame = None
        #     self.projection: str = None
        #     self.offset: int = None
        #     self.duration: int = None
        #     self.group: int = None
        #     self.chunks_list: List[int]

        # Factors
        # if True:
        #     self._name: str = None
        #     self._projection: str = None
        #     self._tiling: Tiling = None
        #     self._quality: int = None
        #     self._tile: int = None
        #     self._chunk: int = None

    # if True:
    #     @property
    #     def name(self):
    #         return self._name
    #
    #     @name.setter
    #     def name(self, name: str):
    #         self._name = name
    #         info = self.config.videos_list[name]
    #         self.original = Path(info['original'])
    #         self.scale = Frame(info['scale'], Position)
    #         self.projection = str(info['projection'])
    #         self.offset = int(info['offset'])
    #         self.duration = int(info['duration'])
    #         self.group = int(info['group'])

    # if True:
    #     @property
    #     def projection(self):
    #         return self.projection
    #
    #     @projection.setter
    #     def projection(self, projection: str):
    #         self._projection = projection
    #
    # if True:
    #     @property
    #     def tiling(self):
    #         return self._tiling
    #
    #     @tiling.setter
    #     def tiling(self, tiling: str):
    #         self._tiling = Tiling(tiling, self.frame)
    #
    # if True:
    #     @property
    #     def quality(self):
    #         return self._quality
    #
    #     @quality.setter
    #     def quality(self, quality: int):
    #         self._quality = quality
    #
    # if True:
    #     @property
    #     def tile(self):
    #         return self._tile
    #
    #     @tile.setter
    #     def tile(self, tile: str):
    #         self._tile = tile
    #
    # if True:
    #     @property
    #     def chunk(self):
    #         return self._chunk
    #
    #     @chunk.setter
    #     def chunk(self, chunk: str):
    #         self._chunk = chunk


# class VideoState(AbstractVideoState):
#     def __init__(self, config: Config, deep: int):
#         """
#         Class to create tile files path to process.
#         :param config: Config object.
#         """
        # self.deep = deep
        # self.config = config
        # self.project = Path(f'results/{config.project}')
        # self.scale = config.scale
        # self.frame = Frame(config.scale)
        # self.fps = config.fps
        # self.gop = config.gop
        # self.rate_control = config.rate_control
        # self.projection = config.projection
        # self.videos_dict = config.videos_list
        #
        # self.videos_list = config.videos_list
        # self.quality_list = config.quality_list
        # self.pattern_list = config.pattern_list
        #
        # self._original_folder = Path(config.original_folder)
        # self._lossless_folder = Path(config.lossless_folder)
        # self._compressed_folder = Path(config.compressed_folder)
        # self._segment_folder = Path(config.segment_folder)
        # self._dectime_folder = Path(config.dectime_folder)
        # self._siti_folder = Path(config.siti_folder)
        # self._check_folder = Path(config.check_folder)
    #
    # def __iter__(self):
    #     count = 0
    #     deep = self.deep
    #     if deep == 0:
    #         count += 1
    #         yield count
    #         return None
    #
    #     for self.video in self.videos_list:
    #         if deep == 1:
    #             count += 1
    #             yield count
    #             continue
    #         for self.tiling in self.tiling_list:
    #             if deep == 2:
    #                 count += 1
    #                 yield count
    #                 continue
    #             for self.quality in self.quality_list:
    #                 if deep == 3:
    #                     count += 1
    #                     yield count
    #                     continue
    #                 for self.tile in self.tiles_list:
    #                     if deep == 4:
    #                         count += 1
    #                         yield count
    #                         continue
    #                     for self.chunk in self.chunk_list:
    #                         if deep == 5:
    #                             count += 1
    #                             yield count
    #                             continue
    #


class TileDecodeBenchmark:
    config: Config = None
    state: VideoContext = None
    role: Role = None

    Role.PREPARE = Operation('PREPARE', 1, 'stub', 'prepare', 'stub')
    Role.COMPRESS = Operation('COMPRESS', 4, 'stub', 'compress', 'stub')
    Role.SEGMENT = Operation('SEGMENT', 4, 'stub', 'segment', 'stub')
    Role.DECODE = Operation('DECODE', 5, 'stub', 'decode', 'stub')
    Role.COLLECT_RESULTS = Operation('COLLECT_RESULTS', 5,
                                     'init_collect_dectime', 'collect_dectime',
                                     'save_dectime')
    Role.SITI = Operation('SITI', 4, 'init_siti', 'calculate_siti', 'end_siti')

    # todo: fazer as variáveis config, role e state parâmetros que verificam se
    #  é None. Serão atributos que não podem ser alterados uma vez que são
    #  definidos.

    def __init__(self, config: str = None, role: str = None, **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        self.results = AutoDict()
        self.results_dataframe = pd.DataFrame()

        if self.config is None:
            self.config = Config(config)
        if self.role is None:
            self.role = Role(role)
        if self.state is None:
            self.state = VideoContext(self.config, self.role)

        self.print_resume()
        self.run(**kwargs)

    def run(self, **kwargs):
        init = self.role.init(self)
        operation = self.role.operation(self)
        finish = self.role.finish(self)

        init()

        total = len(self.state)
        for n in self.state:
            print(f'{n}/{total}', end='\r')
            info(f'\n{self.state}')
            action = operation(**kwargs)

            if action == 'continue':
                continue
            elif action is None:
                break
            elif len(action) == 2:
                fun, params = action
                fun(*params)
        finish()
        print(f'The end of {self.role.name}')

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.role.op.name}\n'
              f'  project: {self.state.project}\n'
              f'  projection: {self.state.projection}\n'
              f'  codec: {self.state.codec}\n'
              f'  fps: {self.state.fps}\n'
              f'  gop: {self.state.gop}\n'
              f'  qualities: {self.state.quality_list}\n'
              f'  patterns: {self.state.tiling_list}'
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
            return 'continue'

        if not original.exists():
            warning(f'The file {original} not exist. Skipping.')
            return 'continue'

        video = self.state
        fps = self.state.fps
        frame = self.state.frame
        dar = frame.w / frame.h

        cmd = f'ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {video.offset} '
        cmd += f'-i {original} '
        cmd += f'-crf 0 '
        cmd += f'-t {video.duration} '
        cmd += f'-r {fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf "scale={frame.resolution},setdar={dar}" '
        cmd += f'{uncompressed_file}'

        return run_command, (cmd, lossless_log, 'w')

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
            return 'continue'

        if not uncompressed_file.exists():
            warning(f'The file {uncompressed_file} not exist. Skipping.')
            return 'continue'

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
                f'w={tile.frame.w}:h={tile.frame.h}:'
                f'x={tile.frame.x}:y={tile.frame.y}"']
        cmd += [f'{compressed_file}']
        cmd = ' '.join(cmd)

        return run_command, (cmd, compressed_log, 'w')

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
            return 'continue'

        if not compressed_file.is_file():
            warning(f'The file {compressed_file} not exist. Skipping.')
            return 'continue'

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
                return 'continue'

        if not segment_file.is_file():
            warning(f'The file {segment_file} not exist. Skipping.')
            return 'continue'

        cmd = [f'ffmpeg -hide_banner -benchmark '
               f'-codec hevc -threads 1 '
               f'-i {segment_file} '
               f'-f null -']

        return self.run_times, (diff, run_command, (cmd, dectime_log, 'a'))

    @staticmethod
    def run_times(num, runner, args):
        for i in range(num):
            runner(*args)

    # COLLECT RESULTS
    def init_collect_dectime(self) -> Any:
        dectime_json_file = self.state.dectime_json_file

        if dectime_json_file.exists():
            dectime_json = json.loads(
                dectime_json_file.read_text(encoding='utf-8'))
            self.results = AutoDict(dectime_json)

    def collect_dectime(self, overwrite=False) -> Any:
        """
        The result dict have a following structure:
        results[video_name][tile_pattern][quality][idx][chunk_id]
                ['utime'|'bit rate']['psnr'|'qp_avg']
        [video_name]    : The video name
        [projection]    : The video projection
        [tile_pattern]  : The tile tiling. eg. "6x4"
        [quality]       : Quality. A int like in crf or qp.
        [idx]           : the tile number. ex. max = 6*4
        if [chunk_id]   : A id for chunk. With 1s chunk, 60s video have 60
                          chunks
            [type]      : "utime" (User time), or "bit rate" (Bit rate in kbps)
                          of a chunk.
        if ['psnr']     : the ffmpeg calculated psnr for tile (before
                          segmentation)
        if ['qp_avg']   : The ffmpeg calculated average QP for a encoding.
        :param overwrite:
        :return:
        """
        debug(f'Collecting {self.state}')

        results = self.results
        for factor in self.state.factors_list:
            results = results[factor]

        if not results == {} and not overwrite:
            warning(
                f'The result key for {self.state} contain some value. '
                f'Skipping.')
            return 'continue'  # if value exist and not overwrite, then skip

        if not self.state.segment_file.exists():
            warning(f'The file {self.state.segment_file} not exist. Skipping.')
            return 'continue'

        if self.state.chunk == 1 and not self.state.compressed_file.exists():
            warning(
                f'The file {self.state.compressed_file} not exist. Skipping.')
            return 'continue'

        try:
            chunk_size = self.state.segment_file.stat().st_size
        except (FileNotFoundError, PermissionError):
            warning(
                f'unexpected error on reading size of '
                f'{self.state.segment_file}. Skipping.')
            return 'continue'

        times = self.get_times()

        bitrate = chunk_size * 8 / (self.state.gop / self.state.fps)

        data = {'bitrate': bitrate, 'dectimes': times}
        results.update(data)
        return 'continue'

    def save_dectime(self, compact=True):
        filename = self.state.dectime_json_file
        info(f'Saving {filename}')
        separators, indent = ((',', ':'), None) if compact else (None, 2)
        json_dumps = json.dumps(self.results, separators=separators,
                                indent=indent)
        filename.write_text(json_dumps, encoding='utf-8')
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

    # 'CALCULATE SITI'
    def init_siti(self):
        self.state.tiling_list = ['1x1']
        self.state.quality_list = [28]
        self.compress(overwrite=False)

    def siti(self, overwrite=False, animate_graph=False, save=True) -> None:
        siti = SiTi(self.state)
        siti.calc_siti(animate_graph=animate_graph, overwrite=overwrite,
                       save=save)

    def end_siti(self):
        self._join_siti()
        self._scatter_plot_siti()

    def _join_siti(self):
        siti_results_final = pd.DataFrame()
        siti_stats_json_final = {}
        num_frames = None

        for name in enumerate(self.state.names_list):
            """Join siti_results"""
            siti_results_file = self.state.siti_results
            siti_results_df = pd.read_csv(siti_results_file)
            if num_frames is None:
                num_frames = self.state.duration * self.state.fps
            elif num_frames < len(siti_results_df['si']):
                dif = len(siti_results_df['si']) - num_frames
                for _ in range(dif):
                    siti_results_df.loc[len(siti_results_df)] = [0, 0]

            siti_results_final[f'{name}_ti'] = siti_results_df['si']
            siti_results_final[f'{name}_si'] = siti_results_df['ti']

            """Join stats"""
            siti_stats = self.state.siti_stats
            with open(siti_stats, 'r', encoding='utf-8') as f:
                individual_stats_json = json.load(f)
                siti_stats_json_final[name] = individual_stats_json
        # siti_results_final.to_csv(f'{self.state.siti_folder /
        # "siti_results_final.csv"}', index_label='frame')
        # pd.DataFrame(siti_stats_json_final).to_csv(f'{self.state.siti_folder
        # / "siti_stats_final.csv"}')

    def _scatter_plot_siti(self):
        siti_results_df = pd.read_csv(
            f'{self.state.siti_folder / "siti_stats_final.csv"}', index_col=0)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
        fig: plt.Figure
        ax: plt.Axes
        for column in siti_results_df:
            si = siti_results_df[column]['si_2q']
            ti = siti_results_df[column]['ti_2q']
            name = column.replace('_nas', '')
            ax.scatter(si, ti, label=name)
        ax.set_xlabel("Spatial Information", fontdict={'size': 12})
        ax.set_ylabel('Temporal Information', fontdict={'size': 12})
        ax.set_title('Si/Ti', fontdict={'size': 16})
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  fontsize='small')
        fig.tight_layout()
        fig.savefig(self.state.siti_folder / 'scatter.png')
        fig.show()


class CheckTileDecodeBenchmark(TileDecodeBenchmark):
    Role.CHECK_ORIGINAL = Operation('CHECK_ORIGINAL', 1, 'stub',
                                    'check_original', 'save')
    Role.CHECK_LOSSLESS = Operation('CHECK_LOSSLESS', 1, 'stub',
                                    'check_lossless', 'save')
    Role.CHECK_COMPRESS = Operation('CHECK_COMPRESS', 4, 'stub',
                                    'check_compress', 'save')
    Role.CHECK_SEGMENT = Operation('CHECK_SEGMENT', 5, 'stub', 'check_segment',
                                   'save')
    Role.CHECK_DECODE = Operation('CHECK_DECTIME', 5, 'stub', 'check_dectime',
                                  'save')
    Role.CHECK_RESULTS = Operation('CHECK_RESULTS', 5, 'load_results',
                                   'check_results', 'save')
    Role.CLEAN = Operation('CLEAN', 0, 'none', 'clean', 'none')
    check_table: Dict[str, list]

    def __init__(self, config: str, role: str, **kwargs):
        self.check_table = {'file': [], 'msg': []}
        try:
            super().__init__(config, role, **kwargs)
        except KeyboardInterrupt:
            self.save()
            raise KeyboardInterrupt

    def check_original(self, **kwargs):
        original_file = self.state.original_file
        debug(f'==== Checking {original_file} ====')
        msg = self.check_video(original_file, **kwargs)
        return self.update_table, (original_file, msg,)

    def check_lossless(self, only_error=True, **kwargs):
        lossless_file = self.state.lossless_file
        debug(f'Checking the file {lossless_file}')

        duration = self.state.duration
        fps = self.state.fps
        log_pattern = f'frame={duration * fps:5}'

        msg = self.check_video(lossless_file, log_pattern, **kwargs)
        return self.update_table, (lossless_file, msg,)

    def check_compress(self, only_error=True, **kwargs):
        video_file = self.state.compressed_file
        debug(f'Checking the file {video_file}')

        duration = self.state.duration
        fps = self.state.fps
        log_pattern = f'encoded {duration * fps} frames'

        msg = self.check_video(video_file, log_pattern=log_pattern, **kwargs)

        if only_error:
            if msg in 'log_ok-video_ok':
                return 'continue'

        return self.update_table, (video_file, msg,)

    def check_segment(self, only_error=True, **kwargs):
        segment_file = self.state.segment_file
        debug(f'Checking the file {segment_file}')

        msg = self.check_video(segment_file, **kwargs)

        if 'video_ok' in msg and only_error:
            return 'continue'

        return self.update_table, (segment_file, msg)

    def check_dectime(self, only_error=True, clean=False):
        dectime_log = self.state.dectime_log
        debug(f'Checking the file {dectime_log}')

        if not dectime_log.exists():
            warning('logfile_not_found')
            count_decode = 0
        else:
            count_decode = self.count_decoding()
            if count_decode == 0:
                dectime_log.unlink()
            if only_error and count_decode >= self.state.decoding_num:
                return 'continue'

        msg = f'decoded_{count_decode}x'
        return self.update_table, (dectime_log, msg,)

    def load_results(self):
        dectime_json_file = self.state.dectime_json_file
        with dectime_json_file.open() as f:
            self.results = json.load(f)

    def check_result(self, only_error=True, clean=False):
        results = self.results
        for factor in self.state.factors_list:
            results = results[factor]

        bitrate = float(results['bitrate'])
        if bitrate > 0:
            msg = 'bitrate_ok'
        else:
            msg = 'bitrate==0'

        dectimes = results['dectimes']
        if len(dectimes) > 0:
            msg += '_dectimes_ok'
        else:
            msg += '_dectimes==0'

        if only_error and msg == 'bitrate_ok_dectimes_ok':
            'continue'

        return self.update_table, (msg,)

    def update_table(self, file, msg):
        self.check_table['file'].append(file)
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

    def clean(self, table_of_check: str, clean_log, clean_video,
              video_of_log=False):
        folder = self.state.check_folder
        table_filename = folder / table_of_check
        check_table = pd.read_csv(table_filename, index_col=0)
        for idx, line in check_table.iterrows():
            video = Path(line['file'])
            msg = line['msg']
            msg_log, msg_video = msg.split('-')
            if msg_log not in 'log_ok' and clean_log:
                log = video.with_suffix('.log')
                debug(f'Cleaning {log}')
                log.unlink(missing_ok=True)
                if video_of_log:
                    video.unlink(missing_ok=True)
            if msg_video not in 'video_ok' and clean_video:
                debug(f'Cleaning {video}')
                video.unlink(missing_ok=True)
        return 'continue'

    def save(self):
        folder = self.state.check_folder
        table_filename: Union[
            Path, None] = folder / (f'{self.role.name}-table-'
                                    f'{datetime.datetime.today()}.csv')
        resume_filename: Union[
            Path, None] = folder / (f'{self.role.name}-resume-'
                                    f'{datetime.datetime.today()}.csv')

        check_table = pd.DataFrame(self.check_table)
        # noinspection PyTypeChecker
        check_table.to_csv(f'{table_filename}'.replace(':', '-'),
                           encoding='utf-8', index_label='counter')

        resume = dict(Counter(check_table['msg']))
        print('Resume:')
        print(json.dumps(resume, indent=2))
        resume = pd.DataFrame.from_dict(resume, orient='index',
                                        columns=('count',))
        # noinspection PyTypeChecker
        resume.to_csv(f'{resume_filename}'.replace(':', '-'), encoding='utf-8',
                      index_label='msg')


HCSPoint = NamedTuple('HCSPoint', (('yaw', float), ('pitch', float)))


class QualityAssessment(TileDecodeBenchmark):
    PIXEL_MAX = 255
    metrics_reg = {'PSNR': 'psnr',
                   'WS-PSNR': 'wspsnr',
                   # 'S-PSNR': 'spsnr_nn',
                   }

    class QualityRole(Role):
        Role.QUALITY_ALL = Operation('QUALITY_ALL', 4, 'all_init', 'all',
                                     'None')
        Role.RESULTS = Operation('RESULTS', 4, 'init_result', 'result',
                                 'save_result')

    def __init__(self, config: str, role: str, sphere_file: str = None,
                 **kwargs):
        self.sph_points: Union[list, None] = None
        self.sph_points_img: Union[list, None] = None
        self.weight_ndarray: Union[np.ndarray, None] = None
        self.metrics_methods: Dict[str, Callable] = {}

        self.sph_file = Path('lib/sphere_655362.txt') \
            if sphere_file is None else Path(sphere_file)
        assert self.sph_file.exists()

        super().__init__(config, role, **kwargs)

    def load_sph_point(self):
        # Load 655362 sample points (elevation, azimuth). Angles in degree.
        content = self.sph_file.read_text().splitlines()[1:]
        for line in content:
            point = list(map(float, line.strip().split()))
            self.sph_points.append(HCSPoint(point[1], point[0]))
            # m, n = sph2erp(np.deg2rad(point[1]), np.deg2rad(point[0]),
            #                self.state.frame.shape)
            # self.sph_points_img.append((m, n))

        # Convert spherical coordinate into Cartesian 3D coordinate
        # cart_coord = []
        # for phi, theta in self.sph_points:
        #     cart_coord.append([np.sin(theta) * np.cos(phi),
        #                        np.sin(phi),
        #                        -np.cos(theta) * np.cos(phi)])
        # #Convert Cartesian 3D coordinate into rectangle coordinate
        # #phi = math.acos(Y), theta = math.atan2(X, Z)
        # rect_coord = []
        # for x in cart_coord:
        #     rect_coord.append([width * (0.5 + np.atan2(x[0], x[2]) / (np.pi *
        #     2)),
        #                        height * (np.acos(x[1]) / np.pi)])

    def all_init(self):
        self.load_sph_point()
        for metric in self.metrics_reg:
            method = self.metrics_reg[metric]
            self.metrics_methods[metric] = eval(f'self.{method}')

    def all(self, overwrite=False):
        debug(f'Processing {self.state}')

        if self.state.quality == self.state.original_quality:
            return 'continue'

        compressed_reference = self.state.reference_file
        compressed_file = self.state.compressed_file
        compressed_quality_csv = self.state.quality_csv

        if compressed_quality_csv.exists() and not overwrite:
            warning(f'The file {compressed_quality_csv} exist. Skipping.')
            return 'continue'

        frames = zip(get_frame(compressed_reference),
                     get_frame(compressed_file))

        results = defaultdict(list)
        for n, (frame_video1, frame_video2) in enumerate(frames):
            debug(f'Frame {n}')
            for metric in self.metrics_methods:
                metrics_method = self.metrics_methods[metric]
                metric_value = metrics_method(frame_video1, frame_video2)
                results[metric].append(metric_value)
        pd.DataFrame(results).to_csv(compressed_quality_csv, encoding='utf-8',
                                     index_label='frame')
        return 'continue'

    def init_result(self):
        compressed_quality_result_json = self.state.quality_result_json
        if compressed_quality_result_json.is_file():
            warning(f'The file {compressed_quality_result_json} exist. '
                    f'Loading.')
            self.results = json.loads(
                compressed_quality_result_json.read_text(encoding='utf-8'))

    def result(self, overwrite=False):
        if self.state.quality == self.state.original_quality:
            return 'continue'

        debug(f'Processing {self.state}')
        compressed_quality_csv = self.state.quality_csv

        if not compressed_quality_csv.is_file():
            warning(f'The file {compressed_quality_csv} not exist. Skipping.')
            return 'continue'

        quality = pd.read_csv(compressed_quality_csv, index_col=0)

        results = self.results
        for factor in self.state.factors_list:
            results = results[factor]

        for metric in self.metrics_reg:
            if results[metric] != {} and not overwrite:
                warning(
                    f'The key [{self.state}][{metric}] exist. Skipping.')
                return 'continue'

            results[metric] = quality[metric].to_list()
        return 'continue'

    def save_result(self):
        compressed_quality_result_json = self.state.quality_result_json
        compressed_quality_result_pickle = compressed_quality_result_json.with_suffix(
            '.pickle')
        compressed_quality_result_pickle.write_bytes(pickle.dumps(self.results))
        compressed_quality_result_json.write_text(json.dumps(self.results),
                                                  encoding='utf-8')

    @staticmethod
    def psnr(im_ref: np.ndarray, im_deg: np.ndarray,
             im_sal: np.ndarray = None) -> float:
        """
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

        Images must be only one channel (luminance)

        :param im_ref:
        :param im_deg:
        :param im_sal:
        :return:
        """
        im_sqr_err = (im_ref - im_deg) ** 2
        if im_sal is not None:
            im_sqr_err = im_sqr_err * im_sal
        mse = np.average(im_sqr_err)
        return mse2psnr(mse)

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

    def wspsnr(self, im_ref: np.ndarray, im_deg: np.ndarray,
               im_sal: np.ndarray = None):
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :param im_sal:
        :return:
        """
        if self.weight_ndarray is None:
            if self.state.projection == 'equirectangular':
                height, width = self.state.frame.resolution.shape
                func = lambda y, x: np.cos(
                    (y + 0.5 - height / 2) * np.pi / height)
                self.weight_ndarray: Union[
                    np.ndarray, object] = np.fromfunction(func, (height, width),
                                                          dtype='float32')
            elif self.state.projection == 'cubemap':
                face = self.state.frame.resolution.shape[
                           0] / 2  # each face must be square (frame aspect ration =3:2).
                radius = face / 2
                squared_distance = lambda y, x: (x + 0.5 - radius) ** 2 + (
                            y + 0.5 - radius) ** 2
                func = lambda y, x: (1 + squared_distance(y, x) / (
                            radius ** 2)) ** (-3 / 2)
                weighted_face = np.fromfunction(func, (int(face),
                                                       int(face)),
                                                dtype='float32')
                weight_ndarray = np.concatenate((weighted_face, weighted_face))
                self.weight_ndarray = np.concatenate(
                    (weight_ndarray, weight_ndarray, weight_ndarray), axis=1)

        x1 = self.state.tile.frame.x
        x2 = self.state.tile.frame.x + self.state.tile.frame.w
        y1 = self.state.tile.frame.y
        y2 = self.state.tile.frame.y + self.state.tile.frame.h
        weight_ndarray = self.weight_ndarray[y1:y2, x1:x2]

        im_weighted = weight_ndarray * (im_ref - im_deg) ** 2

        if im_sal is not None:
            im_weighted = im_weighted * im_sal
        wmse = np.average(im_weighted)

        if wmse == 0:
            return 1000

        return mse2psnr(wmse)

    def spsnr_nn(self, im_ref: np.ndarray, im_deg: np.ndarray,
                 im_sal: np.ndarray = None):
        if self.sph_points_img is None:
            self.sph_points_img = [sph2erp(theta, phi, im_ref.shape) for
                                   phi, theta in self.sph_points]

        sqr_err_salient = []
        for m, n in self.sph_points_img:
            sqr_diff = (im_ref[n - 1, m - 1] - im_deg[n - 1, m - 1]) ** 2

            if im_sal is not None:
                sqr_diff = im_sal[n - 1, m - 1] * sqr_diff

            sqr_err_salient.append(sqr_diff)
        mse = np.average(sqr_err_salient)
        return mse2psnr(mse)

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

