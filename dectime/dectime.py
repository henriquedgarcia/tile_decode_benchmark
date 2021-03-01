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

    class Check(Enum):
        ORIGINAL = 0
        LOSSLESS = 1
        COMPRESSED = 2
        SEGMENT = 3
        DECTIME = 4

    def __init__(self, config_file,
                 automate=False):
        self.config = util.Config(config_file)
        self.state = VideoState(config=self.config)

        self.role: Union[CheckDectime.Check, None] = None
        self.error_df: Union[dict, pd.DataFrame] = dict(video=[], msg=[])

        self.configure()
        if automate is True:
            self.check()
            self.save_report()

    def configure(self) -> None:
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

    def check(self):
        for self.state.video in self.state.videos_list:
            if self.role is self.Check.ORIGINAL:
                self.check_video_state(self.state.original_file)
                continue
            if self.role is self.Check.LOSSLESS:
                self.check_video_state(self.state.lossless_file)
                continue

            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        if self.role is self.Check.COMPRESSED:
                            self.check_video_state(self.state.compressed_file)
                            continue

                        for self.state.chunk in self.state.video.chunks:
                            if self.role is self.Check.SEGMENT:
                                self.check_video_state(self.state.segment_file)
                                continue
                            if self.role is self.Check.DECTIME:
                                self.check_video_state(self.state.dectime_log)
                                continue

        self.error_df = pd.DataFrame(self.error_df)
        print(f'RESUMO: {self.role.name}')
        msg = self.error_df['msg']  # a Pandas Serie
        print(json.dumps(collections.Counter(msg), indent=2))

    def check_video_state(self, video_file) -> None:
        print(f'Checking {video_file}')

        try:
            filesize = os.path.getsize(f'{video_file}')
            if filesize == 0:
                msg = f'size_0'
            else:
                if self.role is self.Check.DECTIME:
                    with open(video_file, 'r', encoding='utf-8') as f:
                        count_decode = 0
                        for line in f:
                            if 'utime' in line: count_decode += 1
                        msg = (f'decoded_{count_decode}_times'
                               if count_decode > 0 else 'decode_error')
                elif self.role is self.Check.COMPRESSED:
                    logfile = f'{video_file[:-4]}.log'
                    try:
                        filesize = os.path.getsize(f'{logfile}')
                        if filesize > 0:
                            with open(logfile, 'r', encoding='utf-8') as f:
                                for line in f:
                                    log = (True if 'Global PSNR' in line
                                           else False)
                                msg = ('apparently ok' if log
                                       else 'encoding_log_error')
                        else:
                            msg = f'encoding_error'
                    except FileNotFoundError:
                        msg = f'encoding_log_not_found'
                else:
                    msg = f'apparently ok'

        except FileNotFoundError:
            msg = f'not_found'

        self.error_df['video'].append(video_file)
        self.error_df['msg'].append(msg)

    def save_report(self, savepath=None):
        if savepath is None:
            savepath = (f"{self.state.project}/check_dectime"
                        f"_{self.role.name}.log")
            os.makedirs(f"{self.state.project}/check_dectime", exist_ok=True)
        self.error_df.to_csv(savepath)
