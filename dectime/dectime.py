import collections
import json
import os
import subprocess
from collections import Counter, defaultdict
from enum import Enum
from os import path
from typing import Union

import numpy as np
import pandas as pd

import dectime.util as util
from dectime.video_state import Tiling, VideoState


class Role(Enum):
    PREPARE = 0
    COMPRESS = 1
    SEGMENT = 2
    DECODE = 3
    RESULTS = 4
    SITI = 5
    ALL = 6


class TileDecodeBenchmark:
    """
    The result dict have a following structure:
    results[video_name][tile_pattern][quality][tile_id][chunk_id][type]
    [video_name]      : The video name
    [tile_pattern]    : The tile pattern. eg. "6x4"
    [quality]         : Quality. A int like in crf or qp.
    [tile_id]         : the tile number. ex. max = 6*4
    [chunk_id or error/distortion]:
        if [chunk_id]     : A id for chunk. With 1s chunk, 60s video have 60 chunks
           [type]         : "utime" (User time), or "bit rate" (Bit rate in kbps) of a chunk.
        if ['psnr']         : the ffmpeg calculated psnr for tile (before segmentation)
        if ['qp_avg']       : The ffmpeg calculated average QP for a encoding.
    """
    role: Role

    def __init__(self, config: str):
        self.config = util.Config(config)
        self.state = VideoState(self.config)
        self.results = util.AutoDict()

    @staticmethod
    def _check_existence(file_path, overwrite=False):
        if path.isfile(f'{file_path}'):
            if overwrite:
                print(f'The file "{file_path}" exist. Overwriting')
                return False
            else:
                print(f'The file "{file_path}" exist. Skipping')
                return True
        return False

    def _collect_dectime(self) -> util.AutoDict:
        """

        :param self:
        :return:
        """
        strip_time = lambda line: float(
                line.strip().split(' ')[1].split('=')[1][:-1])
        with open(self.state.dectime_log, 'r', encoding='utf-8') as f:
            times = [strip_time(line) for line in f
                     if 'utime' in line]
        chunk_size = path.getsize(self.state.segment_file)

        dectime = util.AutoDict()
        dectime['time'] = {'avg': np.average(times),
                           'std': np.std(times)}
        dectime['rate'] = chunk_size * 8 / (self.state.gop / self.state.fps)

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

    def calcule_siti(self, overwrite):
        self.state.quality = 28
        self.state.pattern = Tiling('1x1', self.config.frame)
        self.state.tile = self.state.pattern.tiles_list[0]

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

            siti = util.SiTi(filename=self.state.compressed_file,
                             scale=self.config.frame.scale, plot_siti=False)
            siti.calc_siti(verbose=True)
            siti.save_siti(overwrite=overwrite)
            siti.save_stats(overwrite=overwrite)

    def collect_result(self, overwrite):
        exist = os.path.isfile(self.state.dectime_raw_json)
        if exist and not overwrite:
            exit(f'The file {self.state.dectime_raw_json} exist.')

        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        for self.state.chunk in self.state.video.chunks:
                            print(f'Collecting {self.state.video.name}'
                                  f'-{self.state.pattern.pattern}'
                                  f'-{self.state.factor}{self.state.quality}'
                                  f'-tile{self.state.tile.id}'
                                  f'-chunk{self.state.chunk}')

                            # Collect decode time {avg:float, std:float} and bit rate in bps
                            dectime = self._collect_dectime()
                            self.results[
                                self.state.video.name][
                                self.state.pattern.pattern][
                                self.state.quality][
                                self.state.tile.id][
                                self.state.chunk].update(dectime)
                            # Collect quality List[float]
                            psnr = self._collect_psnr()
                            self.results[
                                self.state.video.name][
                                self.state.pattern.pattern][
                                self.state.quality][
                                self.state.tile.id].update(psnr)
        print(f'Saving {self.state.dectime_raw_json}')
        util.save_json(self.results, self.state.dectime_raw_json, compact=True)

    def prepare_videos(self, overwrite=False) -> None:
        """
        Prepare video to encode.
        :param overwrite:
        """
        for self.state.video in self.state.videos_list:
            uncompressed_file = self.state.lossless_file
            if self._check_existence(uncompressed_file, overwrite): return

            scale = self.state.frame.scale
            fps = self.state.fps
            original = self.state.original_file
            duration = self.state.video.duration
            offset = self.state.video.offset

            command = f'ffmpeg '
            command += f'-y -ss {offset} '
            command += f'-i {original} '
            command += f'-t {duration} -r {fps} -map 0:v -crf 0 '
            command += f'-vf scale={scale},setdar=2 '
            command += f'{uncompressed_file}'
            print(command)

            log = f'{uncompressed_file[:-4]}.log'
            with open(log, 'w', encoding='utf-8') as f:
                subprocess.run(command, shell=True, stdout=f,
                               stderr=subprocess.STDOUT)

    def compress(self, overwrite=False):
        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        compressed_file = self.state.compressed_file
                        exist = self._check_existence(compressed_file,
                                                      overwrite)
                        if exist: return

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
                        with open(log, 'a', encoding='utf-8') as f:
                            subprocess.run(cmd, shell=True, stdout=f,
                                           stderr=subprocess.STDOUT)

    def segment(self, overwrite=False):
        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        log = (f'{self.state.segment_folder}/'
                               f'tile{self.state.tile.id}.log')
                        try:
                            size = os.path.getsize(log)
                            if size > 10000 and not overwrite: continue
                        except FileNotFoundError:
                            continue

                        compressed_file = self.state.compressed_file
                        segment_folder = self.state.segment_folder

                        cmd = 'MP4Box '
                        cmd += '-split 1 '
                        cmd += f'{compressed_file} '
                        cmd += f'-out {segment_folder}/'
                        print(cmd)

                        with open(log, 'w', encoding='utf-8') as f:
                            subprocess.run(cmd, shell=True, stdout=f,
                                           stderr=subprocess.STDOUT)

    def decode(self, overwrite):
        decoding_num = self.config.decoding_num
        count_decoding = CheckProject.count_decoding
        for _ in range(decoding_num):
            for self.state.video in self.state.videos_list:
                for self.state.pattern in self.state.pattern_list:
                    for self.state.quality in self.state.quality_list:
                        for self.state.tile in self.state.pattern.tiles_list:
                            for self.state.chunk in self.state.video.chunks:
                                segment_file = self.state.segment_file
                                dectime_file = self.state.dectime_log

                                count = count_decoding(dectime_file)
                                coding_ok = count >= self.config.decoding_num
                                if coding_ok and not overwrite: continue

                                cmd = (f'ffmpeg -hide_banner -benchmark '
                                       f'-codec hevc -threads 1 ')
                                cmd += f'-i {segment_file} '
                                cmd += f'-f null -'
                                print(cmd)
                                process = subprocess.run(cmd, shell=True,
                                                         capture_output=True,
                                                         stderr=subprocess.STDOUT)
                                with open(dectime_file, 'a', encoding='utf-8')\
                                        as f:
                                    f.write(str(process.stdout))

    def run(self, role: Role, overwrite=False):
        self.role = role
        if role is Role.PREPARE: self.prepare_videos(overwrite=overwrite)
        if role is Role.COMPRESS: self.compress(overwrite=overwrite)
        if role is Role.SEGMENT: self.segment(overwrite=overwrite)
        if role is Role.DECODE: self.decode(overwrite=overwrite)
        if role is Role.RESULTS: self.collect_result(overwrite=overwrite)
        if role is Role.SITI: self.calcule_siti(overwrite=overwrite)


class CheckProject:
    class Check(Enum):
        ORIGINAL = 0
        LOSSLESS = 1
        COMPRESSED = 2
        SEGMENT = 3
        DECTIME = 4

    def __init__(self, config_file):
        self.config = util.Config(config_file)
        self.state = VideoState(config=self.config)
        self.role: Union[CheckProject.Check, None] = None
        self.error_df: Union[dict, pd.DataFrame] = defaultdict(list)

        self.menu()

        self.check()
        self.save_report()

    @staticmethod
    def count_decoding(logfile):
        with open(logfile, 'r', encoding='utf-8') as f:
            count_decode = len(['' for line in f if 'utime' in line])
        return count_decode

    @staticmethod
    def _verify_encode_log(video_file):
        logfile = f'{video_file[:-4]}.log'
        try:
            with open(logfile, 'r', encoding='utf-8') as f:
                msg = ['apparently_ok' for line in f if 'Global PSNR' in line]
                if msg: return msg[0]
        except FileNotFoundError:
            msg = 'logfile_not_found'
            return msg

        msg = 'log_corrupt'
        return msg

    def _check_original(self):
        if self.role is not self.Check.ORIGINAL: return
        for self.state.video in self.state.videos_list:
            video_file = self.state.original_file
            msg = self.check_video_state(video_file)
            self.register_df(video_file, msg)

    def _check_lossless(self):
        if self.role is not self.Check.LOSSLESS: return
        for self.state.video in self.state.videos_list:
            video_file = self.state.lossless_file
            msg = self.check_video_state(video_file)
            self.register_df(video_file, msg)

    def _check_compressed(self):
        if self.role is not self.Check.COMPRESSED: return
        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        video_file = self.state.compressed_file
                        msg = self.check_video_state(video_file)
                        self.register_df(video_file, msg)

    def _check_segment(self):
        if self.role is not self.Check.SEGMENT: return
        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        for self.state.chunk in self.state.video.chunks:
                            video_file = self.state.segment_file
                            msg = self.check_video_state(video_file)
                            self.register_df(video_file, msg)

    def _check_dectime(self):
        if self.role is not self.Check.DECTIME: return
        for self.state.video in self.state.videos_list:
            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        for self.state.chunk in self.state.video.chunks:
                            video_file = self.state.dectime_log
                            msg = self.check_video_state(video_file)
                            self.register_df(video_file, msg)

    def check(self):
        self._check_original()
        self._check_lossless()
        self._check_compressed()
        self._check_segment()
        self._check_dectime()

        print(f'RESUMO: {self.role.name}')
        self.error_df = pd.DataFrame(self.error_df)
        pretty_json = json.dumps(Counter(self.error_df['msg'], indent=2))
        print(pretty_json)

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
        if self.role is self.Check.COMPRESSED:
            msg = self._verify_encode_log(video_file)
            return msg
        elif self.role is self.Check.DECTIME:
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
                      '4 - Check dectime\n'
                      ': ', )
        self.role = self.Check(int(c))

    def save_report(self):
        folder = f"{self.state.project}/check_dectime/"
        filename = f'{folder}/{self.role.name}.log'
        os.makedirs(folder, exist_ok=True)
        self.error_df.to_csv(filename)
