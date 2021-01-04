import collections
import json
import os
import subprocess
from enum import Enum
from os import path
from typing import Union

import numpy as np
import pandas as pd

import dectime.util as util
from dectime.video_state import VideoState


class Role(Enum):
    PREPARE = 0
    COMPRESS = 1
    SEGMENT = 2
    DECODE = 3
    RESULTS = 4


class Dectime:
    """
    The result dict have a following structure:
    results[video_name][tile_pattern][quality][tile_id][chunk_id][type]
    [video_name]      : The video name
    [tile_pattern]    : The tile pattern. eg. "6x4"
    [quality]         : Quality. A int like in crf or qp.
    [tile_id]         : the tile number. ex. max = 6*4
    [chunk_id or error/distortion]:
        if [chunk_id]     : A id for chunk. With 1s chunk, 60s video have 60 chunks
           [type]         : "utime" (User time), or "bitrate" (Bit rate in kbps) of a chunk.
        if ['psnr']         : the ffmpeg calculated psnr for tile (before segmentation)
        if ['qp_avg']       : The ffmpeg calculated average QP for a encoding.
    """

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

    def prepare_videos(self, overwrite=False) -> None:
        """
        Prepare video to encode.
        :param overwrite:
        """
        uncompressed_file = self.state.lossless_file
        if self._check_existence(uncompressed_file, overwrite): return

        scale = self.state.frame.scale
        fps = self.state.fps
        original = self.state.original_file
        duration = self.state.video.duration
        offset = self.state.video.offset
        command = dict()

        command['program'] = f'ffmpeg'
        command['par_in'] = f'-y -ss {offset}'
        command['input'] = f'-i {original}'
        command['par_out'] = f'-t {duration} -r {fps} -map 0:v -crf 0'
        command['filter'] = f'-vf scale={scale},setdar=2'
        command['output'] = f'{uncompressed_file}'
        command = " ".join(list(command.values()))
        print(command)

        log = f'{uncompressed_file[:-4]}.log'
        with open(log, 'w', encoding='utf-8') as f:
            subprocess.run(command, shell=True,
                           stderr=subprocess.STDOUT,
                           stdout=f)

    def compress(self, overwrite=False):
        compressed_file = self.state.compressed_file
        if self._check_existence(compressed_file, overwrite): return

        lossless_file = self.state.lossless_file
        quality = self.state.quality
        gop = self.state.gop
        tile = self.state.tile
        command = dict()

        command['program'] = 'ffmpeg'
        command['global_params'] = '-hide_banner -y -psnr'
        command['input'] = f'-i {lossless_file}'
        command['param_out'] = f'-crf {quality} -tune "psnr"'
        command['encoder_opt'] = (f'-c:v libx265 '
                                  f'-x265-params \''
                                  f'keyint={gop}:'
                                  f'min-keyint={gop}:'
                                  f'open-gop=0:'
                                  f'info=0:'
                                  f'scenecut=0\'')
        command['v_filter'] = (f'-vf "'
                               f'crop=w={tile.w}:h={tile.h}:'
                               f'x={tile.x}:y={tile.y}'
                               f'"')
        command['output'] = f'{compressed_file}'
        command = " ".join(list(command.values()))
        print(command)

        log = f'{compressed_file[:-4]}.log'
        with open(log, 'a', encoding='utf-8') as f:
            subprocess.run(command, shell=True, stderr=subprocess.STDOUT,
                           stdout=f)

    def segment(self, overwrite=False):
        log = f'{self.state.segment_folder}/tile{self.state.tile.id}.log'
        if self._check_existence(log, overwrite): return
        compressed_file = self.state.compressed_file
        segment_folder = self.state.segment_folder
        command = dict()

        command['program'] = 'MP4Box'
        command['params'] = '-split 1'
        command['input'] = compressed_file
        command['output'] = f'-out {segment_folder}/'
        command = " ".join(list(command.values()))
        print(command)

        with open(log, 'w', encoding='utf-8') as f:
            subprocess.run(command, shell=True, stderr=subprocess.STDOUT,
                           stdout=f)

    def decode(self):
        segment_file = self.state.segment_file
        dectime_file = self.state.dectime_file
        command = dict()

        command['program'] = 'ffmpeg'
        command['params_in'] = '-hide_banner -benchmark -codec hevc -threads 1'
        command['input'] = f'-i {segment_file}'
        command['output'] = '-f null -'
        command = " ".join(list(command.values()))
        print(command)

        with open(dectime_file, 'a', encoding='utf-8') as f:
            subprocess.run(command, shell=True, stdout=f,
                           stderr=subprocess.STDOUT)

    def collect_result(self):
        print(f'Collecting {self.state.video.name}'
              f'-{self.state.pattern.pattern}'
              f'-{self.state.quality}'
              f'-{self.state.tile.id}'
              f'-{self.state.chunk}')
        # Collect decode time
        self.results[
            self.state.video.name][
            self.state.pattern.pattern][
            self.state.quality][
            self.state.tile.id][
            self.state.chunk].update(self._collect_dectime())

        # Collect quality
        if self.state.chunk == 1:
            self.results[
                self.state.video.name][
                self.state.pattern.pattern][
                self.state.quality][
                self.state.tile.id].update(self._collect_psnr())
        return self.results

    def _collect_dectime(self) -> util.AutoDict:
        """

        :param self:
        :return:
        """
        dectime = util.AutoDict()
        dectime_file = self.state.dectime_file
        segment_file = self.state.segment_file

        get_time = lambda line: \
            float(line.strip().split(' ')[1].split('=')[1][:-1])
        with open(dectime_file, 'r', encoding='utf-8') as f:
            times = [get_time(line) for line in f
                     if 'utime' in line]
        dectime['utime'] = {'avg': np.average(times),
                            'std': np.std(times)}
        chunk_size = path.getsize(segment_file)
        dectime['bitrate'] = chunk_size * 8 / (self.state.gop / self.state.fps)

        return dectime

    def _collect_psnr(self):
        psnr = util.AutoDict('list')

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

    def save_result(self, filename):
        with open(filename, 'w', encoding='utf-8') as fp:
            json.dump(self.results, fp, separators=(',', ':'))

    def run(self, role):
        for self.state.video in self.state.videos_list:
            if role is Role.PREPARE:
                self.prepare_videos()
                continue

            for self.state.pattern in self.state.pattern_list:
                for self.state.quality in self.state.quality_list:
                    for self.state.tile in self.state.pattern.tiles_list:
                        if role is Role.COMPRESS:
                            self.compress()
                            continue
                        if role is Role.SEGMENT:
                            self.segment()
                            continue

                        for self.state.chunk in self.state.video.chunks:
                            if role is Role.DECODE:
                                self.decode()
                                continue
                            if role is Role.RESULTS:
                                self.collect_result()
                                continue
        if role is Role.RESULTS:
            self.save_result(self.state.result_file)


class CheckDectime:
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
                                self.check_video_state(self.state.dectime_file)
                                continue

        self.error_df = pd.DataFrame(self.error_df)
        print('RESUMO:')
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
                        if count_decode > 0:
                            msg = f'decoded_{count_decode}_times'
                        else:
                            msg = f'decode_error'
                elif self.role is self.Check.COMPRESSED:
                    logfile = f'{video_file[:-4]}.log'
                    try:
                        filesize = os.path.getsize(f'{logfile}')
                        if filesize > 0:
                            with open(logfile, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if 'Global PSNR' in line:
                                        log = True
                                if log:
                                    msg = f'apparently ok'
                                else:
                                    msg = f'encoding_log_error'
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
        self.error_df.to_csv(savepath)
