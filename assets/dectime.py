import json
import os
from collections import Counter, defaultdict
from enum import Enum
from logging import warning, info, debug
from pathlib import Path
from subprocess import run
from typing import Union, Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from assets.siti import SiTi
from assets.util import AutoDict, run_command, AbstractConfig
from assets.video_state import AbstractVideoState, Frame


class Role(Enum):
    CHECK_ORIGINAL = 'check_original'
    PREPARE = 'prepare'
    CHECK_PREPARE = 'check_prepare'
    COMPRESS = 'compress'
    CHECK_COMPRESS = 'check_compress'
    SEGMENT = 'segment'
    CHECK_SEGMENT = 'check_segment'
    DECODE = 'decode'
    CHECK_DECODE = 'check_decode'
    COLLECT_RESULTS = 'collect_dectime'
    CHECK_RESULTS = 'check_results'
    SITI = 'calcule_siti'


class Config(AbstractConfig):
    original_folder = 'original'
    lossless_folder = 'lossless'
    compressed_folder = 'compressed'
    segment_folder = 'segment'
    dectime_folder = 'dectime'
    stats_folder = 'stats'
    graphs_folder = "graphs"
    siti_folder = "siti"
    project: str
    error_metric: str
    decoding_num: int
    scale: str
    projection: str
    codec: str
    fps: int
    gop: int
    distributions: List[str]
    rate_control: str
    quality_list: List[int]
    pattern_list: List[str]
    videos_list: Dict[str, Any]
    videos_file: str

    def __init__(self, config):
        super().__init__(config)

        with open(f'config/{self.videos_file}', 'r') as f:
            video_list = json.load(f)
            self.videos_list: Dict[str, Any] = video_list['videos_list']


class VideoState(AbstractVideoState):
    def __init__(self, config: Config):
        """
        Class to create tile files path to process.
        :param config: Config object.
        """
        self.config = config
        self.project = Path(f'results/{config.project}')
        self.scale = config.scale
        self.frame = Frame(config.scale)
        self.fps = config.fps
        self.gop = config.gop
        self.rate_control = config.rate_control
        self.projection = config.projection
        self.videos_dict = config.videos_list

        self.videos_list = config.videos_list
        self.quality_list = config.quality_list
        self.pattern_list = config.pattern_list

        self._original_folder = Path(config.original_folder)
        self._lossless_folder = Path(config.lossless_folder)
        self._compressed_folder = Path(config.compressed_folder)
        self._segment_folder = Path(config.segment_folder)
        self._dectime_folder = Path(config.dectime_folder)
        self._siti_folder = Path(config.siti_folder)


class TileDecodeBenchmark:
    role: str = None
    method: str
    deep: int
    results = AutoDict()
    results_dataframe: pd.DataFrame
    role_list = {'PREPARE': {'method': 'prepare', 'deep': 1},
                 'COMPRESS': {'method': 'compress', 'deep': 4},
                 'SEGMENT': {'method': 'segment', 'deep': 4},
                 'DECODE': {'method': 'decode', 'deep': 5},
                 'COLLECT_RESULTS': {'method': 'collect_dectime', 'deep': 5},
                 'SITI': {'method': 'calcule_siti', 'deep': 1},
                 }

    def __init__(self, config: str, role: str = None, **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        self.config = Config(config)
        self.state = VideoState(self.config)

        if role is not None:
            self.run(role, **kwargs)

    def run(self, role: str, **kwargs):
        self.role = role
        self.method = self.role_list[role]['method']
        self.deep = self.role_list[role]['deep']
        self.print_resume()

        operation = getattr(self, self.method)
        total = len(list(self.iterate(deep=self.deep)))
        self.progressbar = tqdm(operation(**kwargs), total=total)

        for action in self.progressbar:
            self.progressbar.set_description(desc=self.state.state)
            if action == 'break': break
            if action == 'continue': continue

            fun, params = action
            fun(*params)

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  function: {self.method}\n'
              f'  project: {self.config.project}\n'
              f'  projection: {self.config.projection}\n'
              f'  codec: {self.config.codec}\n'
              f'  fps: {self.config.fps}\n'
              f'  gop: {self.config.gop}\n'
              f'  qualities: {self.config.quality_list}\n'
              f'  patterns: {self.config.pattern_list}'
              )
        print('=' * 70)

    def iterate(self, deep):
        count = 0
        for self.state.video in self.state.videos_list:
            if deep == 1:
                count += 1
                yield count
                continue
            for self.state.tiling in self.state.tiling_list:
                if deep == 2:
                    count += 1
                    yield count
                    continue
                for self.state.quality in self.state.quality_list:
                    if deep == 3:
                        count += 1
                        yield count
                        continue
                    for self.state.tile in self.state.tiles_list:
                        if deep == 4:
                            count += 1
                            yield count
                            continue
                        for self.state.chunk in self.state.chunk_list:
                            if deep == 5:
                                count += 1
                                yield count
                                continue

    # PREPARE
    def prepare(self, overwrite=False) -> None:
        """
        Prepare video to encode. Uncompress and unify resolution, frame rate, pixel format.
        :param overwrite: If True, this method overwrite previous files.
        """
        deep = self.role_list[self.role]['deep']
        for _ in self.iterate(deep=deep):
            original = self.state.original_file
            uncompressed_file = self.state.lossless_file
            lossless_log = self.state.lossless_log

            info(f'Processing {uncompressed_file}')

            if uncompressed_file.exists() and not overwrite:
                warning(f'The file {uncompressed_file} exist. Skipping.')
                continue

            if not original.exists():
                warning(f'The file {original} not exist. Skipping.')
                continue

            video = self.state.video
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
            cmd += f'-vf "scale={frame.scale},setdar={dar}" '
            cmd += f'{uncompressed_file}'

            yield run_command, (cmd, lossless_log, 'w')

    # COMPRESS
    def compress(self, overwrite=False) -> None:
        """
        Encode videos using
        :param overwrite:
        :return:
        """
        for _ in self.iterate(deep=self.deep):
            uncompressed_file = self.state.lossless_file
            compressed_file = self.state.compressed_file
            compressed_log = self.state.compressed_log

            info(f'Processing {compressed_file}')

            if compressed_file.is_file() and not overwrite:
                warning(f'The file {compressed_file} exist. Skipping.')
                continue

            if not uncompressed_file.exists():
                warning(f'The file {uncompressed_file} not exist. Skipping.')
                continue

            quality = self.state.quality
            gop = self.state.gop
            tile = self.state.tile

            cmd = ['ffmpeg']
            cmd += ['-hide_banner -y -psnr']
            cmd += [f'-i {uncompressed_file}']
            cmd += [f'-crf {quality} '
                    f'-tune "psnr"']
            cmd += [f'-c:v libx265']
            cmd += [f'-x265-params']
            cmd += [f'"'
                    f'keyint={gop}:'
                    f'min-keyint={gop}:'
                    f'open-gop=0:'
                    f'scenecut=0:'
                    f'info=0'
                    f'"']
            cmd += [f'-vf']
            cmd += [f'"'
                    f'crop='
                    f'w={tile.w}:h={tile.h}:'
                    f'x={tile.x}:y={tile.y}'
                    f'"']
            cmd += [f'{compressed_file}']
            cmd = ' '.join(cmd)

            yield run_command, (cmd, compressed_log, 'w')

    # SEGMENT
    def segment(self, overwrite=False) -> None:
        for _ in self.iterate(deep=self.deep):
            segment_log = self.state.segment_log
            segment_folder = self.state.segment_folder
            compressed_file = self.state.compressed_file

            if segment_log.is_file() and segment_log.stat().st_size > 10000 and not overwrite:
                # If Check segment log size is very small, infers error and overwrite.
                warning(f'The file {segment_log} exist. Skipping.')
                continue

            if not compressed_file.is_file():
                warning(f'The file {compressed_file} not exist. Skipping.')
                continue

            info(f'Queueing {self.state.basename}, tile {self.state.tile_id}')

            # Alternative:
            # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_time 1 -reset_timestamps 1 output%03d.mp4

            cmd = ['MP4Box']
            cmd += ['-split 1']
            cmd += [f'{compressed_file}']
            cmd += [f'-out {segment_folder}{Path("/")}']
            cmd = ' '.join(cmd)
            cmd = f'bash -c "{cmd}"'

            yield run_command, (cmd, segment_log, 'w')

    # DECODE
    def decode(self, overwrite=False) -> None:
        decoding_num = self.config.decoding_num
        for _ in self.iterate(deep=self.deep):
            for _ in range(decoding_num):
                segment_file = self.state.segment_file
                dectime_log = self.state.dectime_log
                count = self._count_decoding()

                if count == -1:
                    warning(f'TileDecodeBenchmark.decode: Error on reading '
                            f'dectime log file: {dectime_log}.')
                    continue
                elif count >= decoding_num and not overwrite:
                    warning(f'{segment_file} is decoded enough. Skipping.')
                    continue
                if not segment_file.is_file():
                    warning(f'The file {segment_file} not exist. Skipping.')
                    continue

                cmd = [f'ffmpeg -hide_banner -benchmark '
                       f'-codec hevc -threads 1 '
                       f'-i {segment_file} '
                       f'-f null -']

                yield run_command, (cmd, dectime_log, 'a')

    def _count_decoding(self) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :return:
        """
        log_file = self.state.dectime_log
        try:
            content = log_file.read_text(encoding='utf-8')
        except FileNotFoundError:
            return 0
        except UnicodeDecodeError:
            return -1

        return len(['' for line in content.splitlines() if 'utime' in line])

    # COLLECT RESULTS
    def collect_dectime(self, overwrite=False) -> None:
        """
        The result dict have a following structure:
        results[video_name][tile_pattern][quality][idx][chunk_id]['utime'|'bit rate']
                                                       ['psnr'|'qp_avg']
        [video_name]    : The video name
        [tile_pattern]  : The tile tiling. eg. "6x4"
        [quality]       : Quality. A int like in crf or qp.
        [idx]           : the tile number. ex. max = 6*4
        if [chunk_id]   : A id for chunk. With 1s chunk, 60s video have 60 chunks
            [type]      : "utime" (User time), or "bit rate" (Bit rate in kbps) of a chunk.
        if ['psnr']     : the ffmpeg calculated psnr for tile (before segmentation)
        if ['qp_avg']   : The ffmpeg calculated average QP for a encoding.
        :param overwrite:
        :return:
        """
        dectime_json_file = self.state.dectime_json_file

        if dectime_json_file.exists() and not overwrite:
            dectime_json = json.loads(dectime_json_file.read_text(encoding='utf-8'))
            self.results = AutoDict(dectime_json)

        strip_time = lambda line: float(line.strip().split(' ')[1].split('=')[1][:-1])

        for _ in self.iterate(deep=5):
            debug(f'Collecting {self.state.state}')

            if not self.state.segment_file.exists():
                warning(f'The file {self.state.segment_file} not exist.Skipping.')
                continue

            if self.state.chunk == 1 and not self.state.compressed_file.exists():
                warning(f'The file {self.state.compressed_file} not exist.Skipping.')
                continue

            try:
                chunk_size = self.state.segment_file.stat().st_size
                content = self.state.dectime_log.read_text(encoding='utf-8').splitlines()
            except Exception:
                break

            bitrate = chunk_size * 8 / (self.state.gop / self.state.fps)
            times = [strip_time(line) for line in content if 'utime' in line]
            data = {'bitrate': bitrate, 'dectimes': times}
            results = self.results
            for factor in self.state.get_factors():
                results = results[factor]
            if not results == {}: return  # if value exist, then skip
            results.update(data)
            yield 'continue'

        self.save_json()

    def append_result(self, data: dict[str, float]):
        name, pattern, quality, tile, chunk = self.state.get_factors()
        results = self.results[name][pattern][quality][tile][chunk]
        if not results == {}: return  # if value exist, then skip
        results.update(data)

    def save_json(self, compact=True):
        filename = self.state.dectime_json_file
        info(f'Saving {filename}')
        separators, indent = ((',', ':'), None) if compact else (None, 2)
        json_dumps = json.dumps(self.results, separators=separators, indent=indent)
        filename.write_text(json_dumps, encoding='utf-8')

    def json2pd(self):
        """
        old function. Maintained for compatibility
        name_scale_fps_pattern_"CRF"quality_tile| chunk1 | chunk2 | ... | average | std | median

        :return:
        """
        results_dataframe = pd.DataFrame(columns=self.state.chunk_list)
        for _ in self.iterate(deep=4):
            name, pattern, quality, tile, *_ = self.state.get_factors()

            results = self.results[name][pattern][quality][tile]
            chunks_values = [results[chunk] for chunk in self.state.chunk_list]
            results_dataframe.loc[self.state.state] = chunks_values
        self.results_dataframe = pd.DataFrame(results_dataframe)

    # 'CALCULE SITI'
    def siti(self, overwrite=False, animate_graph=False, save=True) -> None:
        self.state.tiling_list = ['1x1']
        self.state.quality_list = [28]
        self.compress(overwrite=False)

        for _ in self.iterate(deep=self.deep):
            siti = SiTi(self.state)
            siti.calc_siti(animate_graph=animate_graph, overwrite=overwrite, save=save)

        self._join_siti()
        self._scatter_plot_siti()

    def _join_siti(self):
        siti_results_final = pd.DataFrame()
        siti_stats_json_final = {}
        num_frames = None

        for _ in enumerate(self.iterate(deep=1)):
            name = self.state.name

            """Join siti_results"""
            siti_results_file = self.state.siti_results
            siti_results_df = pd.read_csv(siti_results_file)
            if num_frames is None:
                num_frames = self.state.video.duration * self.state.fps
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
        siti_results_final.to_csv(f'{self.state.siti_folder / "siti_results_final.csv"}', index_label='frame')
        pd.DataFrame(siti_stats_json_final).to_csv(f'{self.state.siti_folder / "siti_stats_final.csv"}')

    def _scatter_plot_siti(self):
        siti_results_df = pd.read_csv(f'{self.state.siti_folder / "siti_stats_final.csv"}', index_col=0)
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
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')
        fig.tight_layout()
        fig.savefig(self.state.siti_folder / 'scatter.png')
        fig.show()


class CheckTileDecodeBenchmark(TileDecodeBenchmark):
    role_list = {'CHECK_ORIGINAL': {'method': 'check_original', 'deep': 1},
                 'CHECK_PREPARE': {'method': 'check_prepare', 'deep': 1},
                 'CHECK_COMPRESS': {'method': 'check_compress', 'deep': 4},
                 'CHECK_SEGMENT': {'method': 'check_segment', 'deep': 4},
                 'CHECK_DECODE': {'method': 'check_decode', 'deep': 5},
                 'CHECK_RESULTS': {'method': 'check_results', 'deep': 5},
                 }

    def check_prepare(self):
        deep = self.role_list[self.role]['deep']
        for _ in self.iterate(deep=deep):
            lossless_file = self.state.lossless_file

            info(f'Checking the file {lossless_file}')
            msg = self._check_video(lossless_file, check_gop=False)
            debug(f'Message = {msg}')
            return msg

    def check_compress(self):
        video_file = self.state.compressed_file
        info(f'Checking the file {video_file}')

        msg = self._check_video(video_file, check_gop=True)
        debug(f'Message = {msg}')

        return msg

    def check_segment(self):
        video_file = self.state.segment_file
        info(f'Checking the file {video_file}')
        msg = self._check_video(video_file, check_gop=False)
        debug(f'Message = {msg}')
        return msg

    def check_decode(self):
        dectime_log = self.state.dectime_log

        if not dectime_log.exists():
            warning('logfile_not_found')
            return 0

        count_decode = self._count_decoding()

        if count_decode == -1:
            dectime_log.unlink()
            count_decode = 0

        return count_decode

    # 'CHECK VIDEO LOGS AND GOP'
    def _check_video(self, video_file: Path, check_gop) -> str:
        """
        Check video existence, log, size and GOP.
        :param video_file: Path to video
        :param check_gop: must check GOP?
        :return:
        """
        debug(f'Inside _check_video_size method.')
        log = video_file.with_suffix('.log')

        if not video_file.exists():
            log.unlink(missing_ok=True)
            return 'video_not_found'

        if video_file.stat().st_size == 0:
            video_file.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            return 'filesize==0'

        if check_gop:
            info(f'Checking GOP of {video_file}.')
            max_gop, gop = check_video_gop(video_file)[0]
            debug(f'GOP = {gop}')
            debug(f'MaxGOP = {max_gop}')
            if not max_gop == self.config.gop:
                warning(f'Wrong GOP size')
                return f'wrong_gop_size_{max_gop}'
        return 'ok'



        with open(log_file, 'r', encoding='utf-8') as f:
            ok = bool(['' for line in f if 'Global PSNR' in line])

        if not ok:
            self._clean(video_file)
            return 'log_corrupt'

        return 'apparently_ok'

    def _verify_dectime_log(self, dectime_log: Path) -> str:
        if dectime_log.exists():
            count_decode = self._count_decoding(dectime_log)
            if count_decode == -1:
                self._clean(dectime_log)
                msg = f'log_corrupt'
            else:
                msg = f'decoded_{count_decode}x'
        else:
            msg = 'logfile_not_found'
        return msg

    def _clean(self, video_file):
        if self.rem_error:
            debug(f'Deleting {video_file} and the log')
            rem_file(video_file)
            log = video_file.with_suffix('.log')
            rem_file(log)

    @staticmethod
    def check_video_gop(video_file) -> (int, list):
        command = f'ffprobe -show_frames "{video_file}"'
        process = run(command, shell=True, capture_output=True, encoding='utf-8')
        output = process.stdout
        gop = [line.strip().split('=')[1]
               for line in output.splitlines()
               if 'pict_type' in line]

        # Count GOP
        max_gop = 0
        len_gop = 0
        for pict_type in gop:
            if pict_type in 'I':
                len_gop = 1
            else:
                len_gop += 1
            if len_gop > max_gop:
                max_gop = len_gop
        return max_gop, gop




def make_menu(options_txt: list) -> (list, str):
    options = [str(o) for o in range(len(options_txt))]
    menu_lines = ['Options:']
    menu_lines.extend([f'{o} - {text}'
                       for o, text in zip(options, options_txt)])
    menu_lines.append(':')
    menu_txt = '\n'.join(menu_lines)
    return options, menu_txt


def menu(options_txt: list) -> int:
    options, menu_ = make_menu(options_txt)

    c = None
    while c not in options:
        c = input(menu_)

    return int(c)


def check_file_size(video_file) -> int:
    debug(f'Checking size of {video_file}')
    if not os.path.isfile(video_file):
        return -1
    filesize = os.path.getsize(video_file)
    if filesize == 0:
        return 0
    debug(f'The size is {filesize}')
    return filesize


def rem_file(file) -> None:
    if os.path.isfile(file):
        os.remove(file)
