import json
import os
import pickle
from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from logging import warning, info, debug
from pathlib import Path
from subprocess import run
from typing import Union, Any, Dict, List, NamedTuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from assets.siti import SiTi
from assets.util import AutoDict, run_command, AbstractConfig
from assets.video_state import AbstractVideoState, Frame
import skvideo.io
import datetime


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
        self._check_folder = Path('check')


class Operation(NamedTuple):
    name: str
    deep: int
    init: str
    method: str
    finish: str


class Role(ABC):
    NONE = Operation('NONE', 0, 'NONE', 'NONE', 'NONE')

    def __init__(self, name):
        self.op: Operation = getattr(Role, name)

    def init(self, cls) -> Callable:
        try:
            function = getattr(cls, self.op.init)
        except AttributeError:
            function = self.stub
        return function

    def operation(self, cls) -> Callable:
        try:
            function = getattr(cls, self.op.method)
        except AttributeError:
            function = self.stub
        return function

    def finish(self, cls) -> Callable:
        try:
            function = getattr(cls, self.op.finish)
        except AttributeError:
            function = self.stub
        return function

    @property
    def name(self) -> str:
        return self.op.name

    @property
    def deep(self) -> int:
        return self.op.deep

    def stub(self):
        ...


class BaseTileBenchmark:
    config: Config = None
    state: VideoState = None
    role: Role = None

    @abstractmethod
    def __init__(self):
        ...

    def run(self, **kwargs):
        deep = self.role.deep

        init = self.role.init(self)
        operation = self.role.operation(self)
        finish = self.role.finish(self)

        init()

        total = len(list(self.iterate(deep=deep)))
        for n in self.iterate(deep=deep):
            print(f'{n}/{total} - {self.state.state}')
            action = operation(**kwargs)

            if action == 'continue':
                continue
            elif action is None:
                break
            elif len(action) == 2:
                fun, params = action
                fun(*params)
        finish()

    def print_resume(self):
        print('=' * 70)
        print(f'Processing {len(self.config.videos_list)} videos:\n'
              f'  operation: {self.role.op.name}\n'
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
        if deep == 0:
            count += 1
            yield count
            return None

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


class TileDecodeBenchmark(BaseTileBenchmark):
    Role.PREPARE = Operation('PREPARE', 1, 'stub', 'prepare', 'stub')
    Role.COMPRESS = Operation('COMPRESS', 4, 'stub', 'compress', 'stub')
    Role.SEGMENT = Operation('SEGMENT', 4, 'stub', 'segment', 'stub')
    Role.DECODE = Operation('DECODE', 4, 'stub', 'decode', 'stub')
    Role.COLLECT_RESULTS = Operation('COLLECT_RESULTS', 5, 'init_collect_dectime', 'collect_dectime', 'save_dectime')
    Role.SITI = Operation('SITI', 4, 'init_siti', 'calculate_siti', 'end_siti')

    def __init__(self, config: str, role: str, **kwargs):
        """

        :param config:
        :param role: Someone from Role dict
        :param kwargs: Role parameters
        """
        self.results = AutoDict()
        self.results_dataframe: pd.DataFrame

        self.config = Config(config) if self.config is None else self.config
        self.state = VideoState(self.config) if self.state is None else self.state
        self.role = Role(role) if self.role is None else self.role

        self.print_resume()
        self.run(**kwargs)

    # PREPARE
    def prepare(self, overwrite=False) -> Any:
        """
        Prepare video to encode. Uncompress and unify resolution, frame rate, pixel format.
        :param overwrite: If True, this method overwrite previous files.
        """
        original = self.state.original_file
        uncompressed_file = self.state.lossless_file
        lossless_log = self.state.lossless_log

        debug(f'==== Processing {uncompressed_file} ====')

        if uncompressed_file.exists() and not overwrite:
            warning(f'The file {uncompressed_file} exist. Skipping.')
            return 'continue'

        if not original.exists():
            warning(f'The file {original} not exist. Skipping.')
            return 'continue'

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
        compressed_log = self.state.compressed_log

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
                f'w={tile.w}:h={tile.h}:'
                f'x={tile.x}:y={tile.y}"']
        cmd += [f'{compressed_file}']
        cmd = ' '.join(cmd)

        return run_command, (cmd, compressed_log, 'w')

    # SEGMENT
    def segment(self, overwrite=False) -> Any:
        segment_log = self.state.segment_log
        segment_folder = self.state.segment_folder
        compressed_file = self.state.compressed_file

        debug(f'==== Processing {segment_folder} ====')

        if segment_log.is_file() and segment_log.stat().st_size > 10000 and not overwrite:
            # If segment log size is very small, infers error and overwrite.
            warning(f'The file {segment_log} exist. Skipping.')
            return 'continue'

        if not compressed_file.is_file():
            warning(f'The file {compressed_file} not exist. Skipping.')
            return 'continue'

        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_time 1 -reset_timestamps 1 output%03d.mp4

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
        debug(f'==== Processing {dectime_log} ====')

        for _ in range(self.config.decoding_num):
            if self.state.dectime_log.exists():
                content = self.state.dectime_log.read_text(encoding='utf-8')
                count = len(['' for line in content.splitlines()
                             if 'utime' in line])

                if count >= self.config.decoding_num and not overwrite:
                    warning(f'{segment_file} is decoded enough. Skipping.')
                    return 'continue'

            if not segment_file.is_file():
                warning(f'The file {segment_file} not exist. Skipping.')
                return 'continue'

            cmd = [f'ffmpeg -hide_banner -benchmark '
                   f'-codec hevc -threads 1 '
                   f'-i {segment_file} '
                   f'-f null -']

            return run_command, (cmd, dectime_log, 'a')

    # COLLECT RESULTS

    def init_collect_dectime(self) -> Any:
        dectime_json_file = self.state.dectime_json_file

        if dectime_json_file.exists():
            dectime_json = json.loads(dectime_json_file.read_text(encoding='utf-8'))
            self.results = AutoDict(dectime_json)

    def collect_dectime(self, overwrite=False) -> Any:
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
        debug(f'Collecting {self.state.state}')

        results = self.results
        for factor in self.state.get_factors():
            results = results[factor]

        if not results == {} and not overwrite:
            warning(f'The result key for {self.state.state} contain some value. Skipping.')
            return 'continue'  # if value exist and not overwrite, then skip

        if not self.state.segment_file.exists():
            warning(f'The file {self.state.segment_file} not exist. Skipping.')
            return 'continue'

        if self.state.chunk == 1 and not self.state.compressed_file.exists():
            warning(f'The file {self.state.compressed_file} not exist. Skipping.')
            return 'continue'

        try:
            chunk_size = self.state.segment_file.stat().st_size
        except Exception:
            warning(f'unexpected error on reading size of {self.state.segment_file}. Skipping.')
            return 'continue'

        content = self.state.dectime_log.read_text(encoding='utf-8').splitlines()
        strip_time = lambda line: float(line.strip().split(' ')[1].split('=')[1][:-1])
        times = [strip_time(line) for line in content if 'utime' in line]

        bitrate = chunk_size * 8 / (self.state.gop / self.state.fps)

        data = {'bitrate': bitrate, 'dectimes': times}
        results.update(data)
        return 'continue'

    def save_dectime(self, compact=True):
        filename = self.state.dectime_json_file
        info(f'Saving {filename}')
        separators, indent = ((',', ':'), None) if compact else (None, 2)
        json_dumps = json.dumps(self.results, separators=separators, indent=indent)
        filename.write_text(json_dumps, encoding='utf-8')
        # self.json2pd()

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

    # 'CALCULATE SITI'
    def init_siti(self):
        self.state.tiling_list = ['1x1']
        self.state.quality_list = [28]
        self.compress(overwrite=False)

    def siti(self, overwrite=False, animate_graph=False, save=True) -> None:
        siti = SiTi(self.state)
        siti.calc_siti(animate_graph=animate_graph, overwrite=overwrite, save=save)

    def end_siti(self):
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
        # siti_results_final.to_csv(f'{self.state.siti_folder / "siti_results_final.csv"}', index_label='frame')
        # pd.DataFrame(siti_stats_json_final).to_csv(f'{self.state.siti_folder / "siti_stats_final.csv"}')

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
    Role.CHECK_ORIGINAL = Operation('CHECK_ORIGINAL', 1, 'stub', 'check_original', 'save')
    Role.CHECK_LOSSLESS = Operation('CHECK_LOSSLESS', 1, 'stub', 'check_lossless', 'save')
    Role.CHECK_COMPRESS = Operation('CHECK_COMPRESS', 4, 'stub', 'check_compress', 'save')
    Role.CHECK_SEGMENT = Operation('CHECK_SEGMENT', 4, 'stub', 'check_segment', 'save')
    Role.CHECK_DECODE = Operation('CHECK_DECTIME', 5, 'stub', 'check_dectime', 'save')
    Role.CHECK_RESULTS = Operation('CHECK_RESULTS', 5, 'stub', 'check_results', 'save')

    def __init__(self, config: str, role: str, **kwargs):
        self.check_table = {'state': [], 'msg': []}
        super().__init__(config, role, **kwargs)

    def check_original(self, clean=False, check_gop=False):
        original_file = self.state.original_file
        debug(f'==== Checking {original_file} ====')
        log_pattern = None
        msg = self.check_video(original_file, log_pattern, clean=clean, check_gop=check_gop)
        return self.append_msg, (original_file, msg,)

    def check_lossless(self, clean=False, check_gop=False):
        lossless_file = self.state.lossless_file
        debug(f'Checking the file {lossless_file}')
        duration = self.state.video.duration
        fps = self.state.config.fps
        log_pattern = f'frame={duration * fps:5}'
        msg = self.check_video(lossless_file, log_pattern, clean=clean, check_gop=check_gop)

        return self.append_msg, (lossless_file, msg,)

    def check_compress(self, clean=False, check_gop=False):
        video_file = self.state.compressed_file
        debug(f'Checking the file {video_file}')
        duration = self.state.video.duration
        fps = self.state.config.fps
        log_pattern = f'encoded {duration * fps} frames'
        msg = self.check_video(video_file, log_pattern, clean=clean, check_gop=check_gop)

        return self.append_msg, (video_file, msg,)

    def check_segment(self, clean=False, check_gop=False):
        segment_file = self.state.segment_file
        debug(f'Checking the file {segment_file}')
        log_pattern = None
        msg = self.check_video(segment_file, log_pattern, clean=clean, check_gop=check_gop)
        return self.append_msg, (segment_file, msg,)

    def check_dectime(self):
        dectime_log = self.state.dectime_log
        debug(f'Checking the file {dectime_log}')

        if not dectime_log.exists():
            warning('logfile_not_found')
            count_decode = 0
        else:
            count_decode = self.count_decoding()
        msg = f'decoded_{count_decode}x'
        return self.append_msg, (dectime_log, msg,)

    def check_result(self):
        # todo: make it
        dectime_json_file = self.state.dectime_json_file
        debug(f'Checking the file {dectime_json_file}')
        msg = ''
        return self.append_msg, (dectime_json_file, msg,)

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

        return len(['' for line in content if 'utime' in line])

    def append_msg(self, file, msg):
        self.check_table['state'].append(file)
        self.check_table['msg'].append(msg)

    def check_video(self, video: Path, log_pattern, clean=False, check_gop=False) -> str:
        debug(f'Checking video {video}.')
        log = video.with_suffix('.log')
        msg = []

        # Check log
        if log_pattern is None:
            msg += ['log_ok']
        elif not log.exists():
            msg += ['log_not_found']
        elif log.stat().st_size == 0:
            if clean:
                log.unlink(missing_ok=True)
            msg += ['log_size==0']
        else:
            log_content = log.read_text().splitlines()
            encoded_len = len(['' for line in log_content
                               if log_pattern in line])
            if encoded_len > 0:
                msg += ['log_ok']
            else:
                if clean:
                    log.unlink(missing_ok=True)
                msg += ['log_corrupt']

        # Check video
        if not video.exists():
            msg += ['video_not_found']
        elif video.stat().st_size == 0:
            if clean:
                video.unlink(missing_ok=True)
            msg += ['video_size==0']
        else:
            if check_gop:
                max_gop, gop = check_video_gop(video)
                if max_gop != self.config.gop:
                    msg += [f'video_wrong_gop_{max_gop}']
                else:
                    msg += ['video_ok']
            else:
                msg += ['video_ok']
        return '-'.join(msg)

    def save(self):
        folder = self.state.check_folder
        table_filename: Union[Path, None] = folder / f'{self.role.name}-table-{datetime.date.today()}.csv'
        resume_filename: Union[Path, None] = folder / f'{self.role.name}-resume-{datetime.date.today()}.csv'

        check_table = pd.DataFrame(self.check_table)
        check_table.to_csv(table_filename, encoding='utf-8', index_label='counter')

        resume = dict(Counter(check_table['msg']))
        print(json.dumps(resume, indent=2))
        resume = pd.DataFrame.from_dict(resume, orient='index',
                                        columns=('count',))
        resume.to_csv(resume_filename, encoding='utf-8', index_label='msg')


HCSPoint = NamedTuple('HCSPoint', (('yaw', float), ('pitch', float)))
class QualityAssessment(BaseTileBenchmark):
    PIXEL_MAX = 255
    sph_points = []
    sph_points_img: list = None
    weight_ndarray: np.ndarray = None
    metrics_methods: Dict[str, Callable] = {}
    metrics_reg = {'PSNR': 'psnr',
                   'WS-PSNR': 'wspsnr',
                   # 'S-PSNR': 'spsnr_nn',
                   }

    class QualityRole(Role):
        Role.QUALITY_ALL = Operation('QUALITY_ALL', 4, 'all_init', 'all', 'None')
        Role.RESULTS = Operation('RESULTS', 4, 'init_result', 'result', 'save_result')

    def __init__(self, config: str, role: str, sphere_file: str = None, **kwargs):
        self.sph_file = Path('assets/sphere_655362.txt') \
            if sphere_file is None else Path(sphere_file)
        assert self.sph_file.exists()

        self.results = AutoDict()
        self.config = Config(config) if self.config is None else self.config
        self.state = VideoState(self.config) if self.state is None else self.state
        self.role = self.QualityRole(role)

        self.print_resume()
        self.run(**kwargs)

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
        #     rect_coord.append([width * (0.5 + np.atan2(x[0], x[2]) / (np.pi * 2)),
        #                        height * (np.acos(x[1]) / np.pi)])

    def all_init(self):
        # self.load_sph_point()
        for metric in self.metrics_reg:
            method = self.metrics_reg[metric]
            self.metrics_methods[metric] = eval(f'self.{method}')

    def all(self, overwrite=False):
        debug(f'Processing {self.state.state}')

        if self.state.quality == self.state.original_quality:
            return 'continue'

        compressed_reference = self.state.compressed_reference
        compressed_file = self.state.compressed_file
        compressed_quality_csv = self.state.compressed_quality_csv

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
        pd.DataFrame(results).to_csv(compressed_quality_csv, encoding='utf-8', index_label='frame')
        return 'continue'

    def init_result(self):
        compressed_quality_result_json = self.state.compressed_quality_result_json
        if compressed_quality_result_json.is_file():
            warning(f'The file {compressed_quality_result_json} exist. Loading.')
            self.results = json.loads(compressed_quality_result_json.read_text(encoding='utf-8'))

    def result(self, overwrite=False):
        if self.state.quality == self.state.original_quality:
            return 'continue'

        debug(f'Processing {self.state.state}')
        compressed_quality_csv = self.state.compressed_quality_csv

        if not compressed_quality_csv.is_file():
            warning(f'The file {compressed_quality_csv} not exist. Skipping.')
            return 'continue'

        quality = pd.read_csv(compressed_quality_csv, index_col=0)

        results = self.results
        for factor in self.state.get_factors():
            results = results[factor]

        for metric in self.metrics_reg:
            if results[metric] != {} and not overwrite:
                warning(f'The key [{self.state.state}][{metric}] exist. Skipping.')
                return 'continue'

            results[metric] = quality[metric].to_list()
        return 'continue'

    def save_result(self):
        compressed_quality_result_json = self.state.compressed_quality_result_json
        compressed_quality_result_pickle = compressed_quality_result_json.with_suffix('.pickle')
        compressed_quality_result_pickle.write_bytes(pickle.dumps(self.results))
        compressed_quality_result_json.write_text(json.dumps(self.results), encoding='utf-8')

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

    def wspsnr(self, im_ref: np.ndarray, im_deg: np.ndarray, im_sal: np.ndarray = None):
        """
        Must be same size
        :param im_ref:
        :param im_deg:
        :param im_sal:
        :return:
        """
        if self.weight_ndarray is None:
            if self.config.projection == 'equirectangular':
                height, width = self.state.frame.shape
                func = lambda y, x: np.cos((y + 0.5 - height / 2) * np.pi / height)
                self.weight_ndarray: Union[np.ndarray, object] = np.fromfunction(func, (height, width), dtype='float32')
            elif self.config.projection == 'cubemap':
                face = self.state.frame.shape[0] / 2  # each face must be square (frame aspect ration =3:2).
                radius = face / 2
                squared_distance = lambda y, x: (x + 0.5 - radius) ** 2 + (y + 0.5 - radius) ** 2
                func = lambda y, x: (1 + squared_distance(y, x) / (radius ** 2)) ** (-3 / 2)
                weighted_face: Union[np.ndarray, object] = np.fromfunction(func, (int(face), int(face)),
                                                                           dtype='float32')
                weight_ndarray = np.concatenate((weighted_face, weighted_face))
                self.weight_ndarray = np.concatenate((weight_ndarray, weight_ndarray, weight_ndarray), axis=1)

        x1 = self.state.tile.x
        x2 = self.state.tile.x + self.state.tile.w
        y1 = self.state.tile.y
        y2 = self.state.tile.y + self.state.tile.h
        weight_ndarray = self.weight_ndarray[y1:y2, x1:x2]

        im_weighted = weight_ndarray * (im_ref - im_deg) ** 2

        if im_sal is not None:
            im_weighted = im_weighted * im_sal
        wmse = np.average(im_weighted)

        if wmse == 0:
            return 1000

        return mse2psnr(wmse)

    def spsnr_nn(self, im_ref: np.ndarray, im_deg: np.ndarray, im_sal: np.ndarray = None):
        if self.sph_points_img is None:
            self.sph_points_img = [sph2erp(theta, phi, im_ref.shape) for phi, theta in self.sph_points]

        sqr_err_salienced = []
        for m, n in self.sph_points_img:
            sqr_diff = (im_ref[n - 1, m - 1] - im_deg[n - 1, m - 1]) ** 2

            if im_sal is not None:
                sqr_diff = im_sal[n - 1, m - 1] * sqr_diff

            sqr_err_salienced.append(sqr_diff)
        mse = np.average(sqr_err_salienced)
        return mse2psnr(mse)

    def ffmpeg_psnr(self):
        if self.state.chunk == 1:
            name, pattern, quality, tile, chunk = self.state.get_factors()
            results = self.results[name][pattern][quality][tile]
            results.update(self._collect_ffmpeg_psnr())

    def _collect_ffmpeg_psnr(self) -> Dict[str, float]:
        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])
        psnr = None

        content = self.state.compressed_log.read_text(encoding='utf-8')
        content = content.splitlines()

        for line in content:
            if 'Global PSNR' in line:
                psnr = {'psnr': get_psnr(line),
                        'qp_avg': get_qp(line)}
                break
        return psnr


####### erp2sph
def erp2sph(m, n, shape: tuple) -> tuple[int, int]:
    return uv2sph(*erp2uv(m, n, shape))


def uv2sph(u, v):
    theta = (u - 0.5) * 2 * np.pi
    phi = -(v - 0.5) * np.pi
    return phi, theta


def erp2uv(m, n, shape: tuple):
    u = (m + 0.5) / shape[1]
    v = (n + 0.5) / shape[0]
    return u, v


####### sph2erp
def sph2erp(theta, phi, shape: tuple) -> tuple[int, int]:
    return uv2img(*sph2uv(theta, phi), shape)


def sph2uv(theta, phi):
    PI = np.pi
    while True:
        if theta >= PI:
            theta -= 2 * PI
            continue
        elif theta < -PI:
            theta += 2 * PI
            continue
        if phi < -PI / 2:
            phi = -PI - phi
            continue
        elif phi > PI / 2:
            phi = PI - phi
            continue
        break
    u = theta / (2 * PI) + 0.5
    v = -phi / PI + 0.5
    return u, v


def uv2img(u, v, shape: tuple):
    m = round(u * shape[1] - 0.5)
    n = round(v * shape[0] - 0.5)
    return m, n


####### Util
def get_frame(video_path, gray=True, dtype='float32'):
    vreader = skvideo.io.vreader(f'{video_path}', as_grey=gray)
    for frame in vreader:
        if gray:
            _, height, width, _ = frame.shape
            frame = frame.reshape((height, width)).astype(dtype)
        yield frame


def check_video_gop(video_file) -> (int, list):
    command = (f'ffprobe.exe -hide_banner -loglevel 0 '
               f'-of default=nk=1:nw=1 '
               f'-show_entries frame=pict_type '
               f'"{video_file}"')
    process = run(command, shell=True, capture_output=True, encoding='utf-8')
    output = process.stdout
    gop = []
    max_gop = 0
    len_gop = 0
    for line in output.splitlines():
        line = line.strip()
        if line in ['I', 'B', 'P']:
            if line in 'I':
                len_gop = 1
            else:
                len_gop += 1
            if len_gop > max_gop:
                max_gop = len_gop
            gop.append(line)
    return max_gop, gop


def check_file_size(video_file) -> int:
    debug(f'Checking size of {video_file}')
    if not os.path.isfile(video_file):
        return -1
    filesize = os.path.getsize(video_file)
    if filesize == 0:
        return 0
    debug(f'The size is {filesize}')
    return filesize


def mse2psnr(mse: float, pixel_max=255) -> float:
    return 10 * np.log10((pixel_max ** 2 / mse))
