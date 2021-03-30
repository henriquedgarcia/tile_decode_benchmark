import json
import os
from collections import Counter, defaultdict
from enum import Enum
from logging import debug, warning
from os import path
from os.path import exists
from typing import Dict, Union

import numpy as np
import pandas as pd

import assets.util as util
from assets.config import Config
from assets.dectime_types import Check
from assets.siti import SiTi
from assets.util import run_command
from assets.video_state import Tiling, VideoState


class Role(Enum):
    PREPARE = 'prepare_videos'
    COMPRESS = 'compress'
    SEGMENT = 'segment'
    DECODE = 'decode'
    RESULTS = 'collect_result'
    SITI = 'calcule_siti'
    CHECK = 'check'


class TileDecodeBenchmark:
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
    """
    role: Role
    results = util.AutoDict()

    def __init__(self, config: str):
        self.config = Config(config)
        self.state = VideoState(self.config)

    def _iterate(self, deep):
        for self.state.video in self.state.videos_list:
            if deep == 1:
                yield
                continue
            for self.state.tiling in self.state.tiling_list:
                if deep == 2:
                    yield
                    continue
                for self.state.quality in self.state.quality_list:
                    if deep == 3:
                        yield
                        continue
                    for self.state.tile in self.state.tiles_list:
                        if deep == 4:
                            yield
                            continue
                        for self.state.chunk in self.state.chunk_list:
                            if deep == 5:
                                yield
                                continue

    def _collect_dectime(self) -> Dict[str, float]:
        """

        :return:
        """
        chunk_size = path.getsize(self.state.segment_file)
        chunk_size = chunk_size * 8 / (self.state.gop / self.state.fps)

        strip_time = lambda line: float(line.strip().split(' ')[1]
                                        .split('=')[1][:-1])
        with open(self.state.dectime_log, 'r', encoding='utf-8') as f:
            times = [strip_time(line) for line in f if 'utime' in line]

        dectime = {'time'    : np.average(times),
                   'time_std': np.std(times),
                   'rate'    : chunk_size}

        return dectime

    def _collect_psnr(self):
        psnr = util.AutoDict()

        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])

        with open(f'{self.state.compressed_file[:-4]}.log', 'r',
                  encoding='utf-8') as f:
            for line in f:
                if 'Global PSNR' in line:
                    psnr['psnr'] = get_psnr(line)
                    psnr['qp_avg'] = get_qp(line)
                    break
        return psnr

    def run(self, role: str, overwrite=False):
        operation = getattr(self, Role[role].value)
        operation(overwrite=overwrite)

    def prepare_videos(self, overwrite=False) -> None:
        """
        Prepare video to encode.
        :param overwrite:
        """
        for _ in self._iterate(deep=1):
            uncompressed_file = self.state.lossless_file
            if exists(uncompressed_file) and not overwrite:
                print(f'The file {uncompressed_file} exist. Skipping.')
                continue

            print(f'Processing {uncompressed_file}')

            frame = self.state.frame
            fps = self.state.fps
            original = self.state.original_file
            video = self.state.video
            dar = frame.w / frame.h

            command = f'ffmpeg '
            command += f'-y -ss {video.offset} '
            command += f'-i {original} '
            command += f'-t {video.duration} -r {fps} -map 0:v -crf 0 '
            command += f'-vf scale={frame.scale},setdar={dar} '
            command += f'{uncompressed_file}'
            print(command)

            log = f'{path.splitext(uncompressed_file)[0]}.log'
            run_command(command, log)

    def calcule_siti(self, overwrite) -> None:
        self.state.quality = 28
        self.state.tiling = Tiling('1x1', self.state.frame)
        self.state.tile = self.state.tiling.tiles_list[0]

        for self.state.video in self.state.videos_list:
            # Codificar os vídeos caso não estejam codificados.
            compressed_file = self.state.compressed_file
            exist_encoded = os.path.isfile(compressed_file)
            if not exist_encoded or overwrite:
                filename = self.state.compressed_file
                folder, _ = os.path.split(filename)
                _, tail = os.path.split(folder)
                folder = f'{self.state.project}/siti/{tail}'
                os.makedirs(folder, exist_ok=True)
                self.compress(overwrite=overwrite)

            siti = SiTi(filename=self.state.compressed_file,
                        scale=self.state.scale, plot_siti=False)
            siti.calc_siti(verbose=True)
            siti.save_siti(overwrite=overwrite)
            siti.save_stats(overwrite=overwrite)

    def compress(self, overwrite=False):
        for _ in self._iterate(deep=4):
            compressed_file = self.state.compressed_file
            if exists(compressed_file) and not overwrite:
                warning(f'The file {compressed_file} exist. Skipping.')
                continue

            print(f'Processing {compressed_file}')
            # todo: o pattern no basename está saindo errado. Todos são 1x1.

            lossless_file = self.state.lossless_file
            quality = self.state.quality
            gop = self.state.gop
            tile = self.state.tile

            cmd = 'ffmpeg '
            cmd += '-hide_banner -y -psnr '
            cmd += f'-i {lossless_file} '
            cmd += f'-crf {quality} -tune "psnr" '
            cmd += (f'-c:v libx265 '
                    f'-x265-params \''
                    f'keyint={gop}:'
                    f'min-keyint={gop}:'
                    f'open-gop=0:'
                    f'info=0:'
                    f'scenecut=0\' ')
            cmd += (f'-vf "'
                    f'crop=w={tile.w}:h={tile.h}:'
                    f'x={tile.x}:y={tile.y}'
                    f'" ')
            cmd += f'{compressed_file}'
            print(cmd)

            log = f'{compressed_file[:-4]}.log'
            run_command(cmd, log)

    def segment(self, overwrite=False):
        for _ in self._iterate(deep=4):
            # Check segment log size. If size is very small, overwrite.
            log = f'{self.state.segment_folder}/tile{self.state.tile_id}.log'
            try:
                size = os.path.getsize(log)
                if size > 10000 and not overwrite:
                    continue
            except FileNotFoundError:
                pass

            print(f'Processing {self.state.segment_folder}')
            compressed_file = self.state.compressed_file
            segment_folder = self.state.segment_folder

            cmd = 'MP4Box '
            cmd += '-split 1 '
            cmd += f'{compressed_file} '
            cmd += f'-out {segment_folder}/'
            print(cmd)

            run_command(cmd, log)

    def decode(self, overwrite):
        decoding_num = self.config.decoding_num
        for _ in range(decoding_num):
            for _ in self._iterate(deep=5):
                segment_file = self.state.segment_file
                dectime_file = self.state.dectime_log

                count = CheckProject.count_decoding(dectime_file)
                coding_ok = count >= decoding_num
                if coding_ok and not overwrite: continue

                cmd = (f'ffmpeg -hide_banner -benchmark '
                       f'-codec hevc -threads 1 ')
                cmd += f'-i {segment_file} '
                cmd += f'-f null -'
                debug(cmd)

                run_command(cmd, dectime_file, mode='a')

    def collect_result(self, overwrite):
        exist = os.path.isfile(self.state.dectime_raw_json)
        if exist and not overwrite:
            print(f'The file {self.state.dectime_raw_json} exist.')
            return

        for _ in self._iterate(deep=5):
            print(f'Collecting {self.state.name}'
                  f'-{self.state.pattern}'
                  f'-{self.state.factor}{self.state.quality}'
                  f'-tile{self.state.tile_id}'
                  f'-chunk{self.state.chunk}')

            # Collect decode time {avg:float, std:float} and bit rate in bps
            dectime = self._collect_dectime()
            self.results[
                self.state.name][
                self.state.pattern][
                self.state.quality][
                self.state.tile_id][
                self.state.chunk].update(dectime)
            # Collect quality List[float]
            psnr = self._collect_psnr()
            self.results[
                self.state.name][
                self.state.pattern][
                self.state.quality][
                self.state.tile_id].update(psnr)
        print(f'Saving {self.state.dectime_raw_json}')
        util.save_json(self.results, self.state.dectime_raw_json, compact=True)


class CheckProject:
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.state = VideoState(config=self.config)
        self.role: Union[Check, None] = None
        self.error_df: Union[dict, pd.DataFrame] = defaultdict(list)

        self.menu()
        self.check()
        self.save_report()

    @staticmethod
    def count_decoding(log_file):
        try:
            f = open(log_file, 'r', encoding='utf-8')
            count_decode = len(['' for line in f if 'utime' in line])
            f.close()
        except FileNotFoundError:
            count_decode = 0
        return count_decode

    @staticmethod
    def clean(video_file):
        log = CheckProject.get_logfile(video_file)
        os.remove(video_file)
        os.remove(log)

    @staticmethod
    def get_logfile(video_file):
        return f'{video_file[:-4]}.log'

    @staticmethod
    def _verify_encode_log(video_file, rem_error=False):
        log_file = CheckProject.get_logfile(video_file)
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                msg = ['apparently_ok' for line in f if 'Global PSNR' in line]
                if msg:
                    msg = msg[0]
                else:
                    msg = 'log_corrupt'
                    if rem_error:
                        CheckProject.clean(video_file)
        except FileNotFoundError:
            msg = 'logfile_not_found'
        return msg

    def _check_original(self):
        if self.role is not Check.ORIGINAL: return
        for self.state.video in self.state.videos_list:
            video_file = self.state.original_file
            msg = self.check_video_state(video_file)
            self.register_df(video_file, msg)

    def _check_lossless(self):
        if self.role is not Check.LOSSLESS: return
        for self.state.video in self.state.videos_list:
            video_file = self.state.lossless_file
            msg = self.check_video_state(video_file)
            self.register_df(video_file, msg)

    def _check_compressed(self):
        if self.role is not Check.COMPRESSED: return
        for self.state.video in self.state.videos_list:
            for self.state.tiling in self.state.tiling_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.tiles_list:
                        video_file = self.state.compressed_file
                        msg = self.check_video_state(video_file)
                        self.register_df(video_file, msg)

    def _check_segment(self):
        if self.role is not Check.SEGMENT: return
        for self.state.video in self.state.videos_list:
            for self.state.tiling in self.state.tiling_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.tiles_list:
                        for self.state.chunk in self.state.chunk_list:
                            video_file = self.state.segment_file
                            msg = self.check_video_state(video_file)
                            self.register_df(video_file, msg)

    def _check_dectime(self):
        if self.role is not Check.DECTIME: return
        for self.state.video in self.state.videos_list:
            for self.state.tiling in self.state.tiling_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.tiles_list:
                        for self.state.chunk in self.state.chunk_list:
                            video_file = self.state.dectime_log
                            msg = self.check_video_state(video_file)
                            self.register_df(video_file, msg)

    def check(self):
        self._check_original()
        self._check_lossless()
        self._check_compressed()
        self._check_segment()
        self._check_dectime()

        self.error_df = pd.DataFrame(self.error_df)

    def check_video_state(self, video_file) -> str:
        print(f'Checking {video_file}')
        try:
            filesize = os.path.getsize(f'{video_file}')
        except FileNotFoundError:
            msg = f'not_found'
            return msg

        if filesize == 0:
            msg = f'filesize==0'
            return msg

        # if file exist and size > 0
        msg = f'apparently_ok'
        if self.role is Check.COMPRESSED:
            msg = self._verify_encode_log(video_file)
            return msg
        elif self.role is Check.DECTIME:
            count_decode = self.count_decoding(video_file)
            msg = f'decoded_{count_decode}x'
            return msg
        return msg

    def register_df(self, video_file, msg):
        self.error_df['video'].append(video_file)
        self.error_df['msg'].append(msg)

    def menu(self) -> None:
        c = None
        while c not in ['0', '1', '2', '3', '4']:
            c = input('Options:\n'
                      '0 - Check original\n'
                      '1 - Check lossless\n'
                      '2 - Check encoded\n'
                      '3 - Check segments\n'
                      '4 - Check dectime logs\n'
                      ': ', )
        self.role = Check(int(c))

    def save_report(self):
        folder = f"{self.state.project}/check_dectime/"
        os.makedirs(folder, exist_ok=True)

        filename = f'{folder}/{self.role.name}.log'
        self.error_df.to_csv(filename)

        pretty_json = json.dumps(Counter(self.error_df['msg']), indent=2)
        print(f'RESUMO: {self.role.name}\n'
              f'{pretty_json}')

        filename = f'{folder}/{self.role.name}-resume.log'
        with open(filename, 'w') as f:
            json.dump(Counter(self.error_df['msg']), f, indent=2)
