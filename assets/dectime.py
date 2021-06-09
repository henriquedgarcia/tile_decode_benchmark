import json
import os
from collections import Counter, defaultdict
from enum import Enum
from logging import warning, info, debug, critical
from os.path import getsize, isfile, splitext
from pathlib import Path
from subprocess import run
from typing import Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from assets.config import Config
from assets.siti import SiTi
from assets.util import AutoDict, run_command, save_json
from assets.video_state import Tiling, VideoState


class Check(Enum):
    ORIGINAL = 'check_original'
    LOSSLESS = 'check_lossless'
    COMPRESS = 'check_compressed'
    SEGMENT = 'check_segment'
    DECODE = 'check_dectime'


class Role(Enum):
    PREPARE = 'prepare_videos'
    COMPRESS = 'compress'
    SEGMENT = 'segment'
    DECODE = 'decode'
    RESULTS = 'collect_result'
    SITI = 'calcule_siti'
    CHECK = 'check_all'


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
    results = AutoDict()
    log = defaultdict(list)

    def __init__(self, config: str):
        self.config = Config(config)
        self.state = VideoState(self.config)

    def run(self, role: str, **kwargs):
        operation = getattr(self, Role[role].value)
        operation(**kwargs)

    def prepare_videos(self, overwrite=False) -> None:
        """
        Prepare video to encode.
        :param overwrite:
        """
        for _ in self._iterate(deep=1):
            uncompressed_file = self.state.lossless_file
            if isfile(uncompressed_file) and not overwrite:
                warning(f'The file {uncompressed_file} exist. Skipping.')
                continue

            info(f'Processing {uncompressed_file}')

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
            log = f'{splitext(uncompressed_file)[0]}.log'
            run_command(command, log)

    def compress(self, overwrite=False):
        queue = []
        for _ in self._iterate(deep=4):
            compressed_file = self.state.compressed_file
            if isfile(compressed_file) and not overwrite:
                warning(f'The file {compressed_file} exist. Skipping.')
                continue

            info(f'Processing {compressed_file}')

            lossless_file = self.state.lossless_file
            quality = self.state.quality
            gop = self.state.gop
            tile = self.state.tile

            cmd = 'ffmpeg '
            cmd += '-hide_banner -y -psnr '
            cmd += f'-i {lossless_file} '
            cmd += f'-crf {quality} -tune "psnr" '
            cmd += (f'-c:v libx265 '
                    f'-x265-params "'
                    f'keyint={gop}:'
                    f'min-keyint={gop}:'
                    f'open-gop=0:'
                    f'scenecut=0:'
                    f'info=0'
                    f'" ')
            cmd += (f'-vf "'
                    f'crop=w={tile.w}:h={tile.h}:'
                    f'x={tile.x}:y={tile.y}'
                    f'" ')
            cmd += f'{compressed_file}'
            log = self.get_logfile(compressed_file)

            queue.append((cmd, log))

        for cmd in tqdm(queue):
            run_command(*cmd)

    def segment(self, overwrite=False):
        queue = []
        for _ in self._iterate(deep=4):
            # Check segment log size. If size is very small, overwrite.
            segment_folder = self.state.segment_folder
            log = f'{segment_folder}/tile{self.state.tile_id}.log'
            try:
                size = os.path.getsize(log)
                if size > 10000 and not overwrite:
                    warning(f'The segments of "{segment_folder}" exist. Skipping')
                    continue
            except FileNotFoundError:
                pass

            info(f'Queueing {segment_folder}')

            cmd = 'MP4Box '
            cmd += '-split 1 '
            cmd += f'{self.state.compressed_file} '
            cmd += f'-out {segment_folder}{Path("/")}'
            queue.append((cmd, log))

        for cmd in tqdm(queue):
            run_command(*cmd)

    def decode(self, overwrite):
        decoding_num = self.config.decoding_num
        queue = []
        for _ in range(decoding_num):
            for _ in self._iterate(deep=5):
                segment_file = self.state.segment_file
                dectime_file = self.state.dectime_log

                count = CheckProject.count_decoding(dectime_file)
                if count == -1:
                    warning(f'TileDecodeBenchmark.decode: Error on reading '
                            f'dectime log file: {dectime_file}.')
                    continue
                coding_ok = count >= decoding_num
                if coding_ok and not overwrite: continue

                cmd = (f'ffmpeg -hide_banner -benchmark '
                       f'-codec hevc -threads 1 ')
                cmd += f'-i {segment_file} '
                cmd += f'-f null -'

                queue.append((cmd, dectime_file))

            for cmd in tqdm(queue):
                run_command(*cmd, mode='a')

    def collect_result(self, overwrite):
        exist = os.path.isfile(self.state.dectime_raw_json)
        if exist and not overwrite:
            print(f'The file {self.state.dectime_raw_json} exist.')
            return

        for _ in self._iterate(deep=5):
            name, pattern, quality, tile, chunk = self.state.get_factors()
            print(f'Collecting {name}'
                  f'-{pattern}'
                  f'-{self.state.factor}{quality}'
                  f'-tile{tile}'
                  f'-chunk{chunk}')

            # Collect decode time {avg:float, std:float} and bit rate in bps
            self.results[name][pattern][quality][tile][chunk]\
                .update(self._collect_dectime())

            if chunk > 1: continue
            # Collect quality {'psnr': float, 'qp_avg': float}
            self.results[name][pattern][quality][tile]\
                .update(self._collect_psnr())
        print(f'Saving {self.state.dectime_raw_json}')
        save_json(self.results, self.state.dectime_raw_json, compact=True)

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

    def _iterate(self, deep):
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

    def _collect_dectime(self) -> Dict[str, float]:
        """

        :return:
        """
        chunk_size = getsize(self.state.segment_file)
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
        psnr: Dict[str, float] = {}
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

    @staticmethod
    def get_logfile(video_file):
        return f'{video_file[:-4]}.log'


class CheckProject(TileDecodeBenchmark):
    rem_error: bool = None
    error_df: pd.DataFrame = pd.DataFrame(columns=['video', 'msg'])
    role: Union[Check, None] = None

    def check_all(self):
        self.check_original()
        self.save_report()
        self.check_lossless()
        self.save_report()
        self.check_compressed()
        self.save_report()
        self.check_segment()
        self.save_report()
        self.check_dectime()
        self.save_report()

    def run(self, role: str, overwrite=False, rem_error=False):
        self.rem_error = rem_error
        self.role = Check[role]
        getattr(self, self.role.value)()
        self.save_report()

    def check_original(self):
        df = self.error_df
        files_list = []
        for _ in self._iterate(deep=1):
            video_file = self.state.original_file
            files_list.append(video_file)

        # for cmd in tqdm(files_list):

        for i, video_file in enumerate(tqdm(files_list)):
            # sg.one_line_progress_meter('This is my progress meter!', i + 1,
            #                            len(files_list), '-key-')
            msg = self._check_video_size(video_file)
            df.loc[len(df)] = [video_file, msg]
            if i % 299 == 0: self.save_report()

    def check_lossless(self):
        df = self.error_df
        files_list = []
        for _ in self._iterate(deep=1):
            video_file = self.state.lossless_file
            files_list.append(video_file)

        for i, video_file in enumerate(tqdm(files_list)):
            msg = self._check_video_size(video_file)
            df.loc[len(df)] = [video_file, msg]
            if i % 299 == 0: self.save_report()

    def check_compressed(self):
        df = self.error_df
        files_list = []
        for _ in self._iterate(deep=4):
            if _ > 50: break
            video_file = self.state.compressed_file
            files_list.append(video_file)

        for i, video_file in enumerate(tqdm(files_list)):
            msg = self._check_video_size(video_file, check_gop=False)
            if 'ok' in msg:
                msg = self._verify_encode_log(video_file)
            df.loc[len(df)] = [video_file, msg]
            if i % 299 == 0:
                self.save_report()

    def check_segment(self):
        df = self.error_df
        files_list = []
        for _ in self._iterate(deep=5):
            video_file = self.state.segment_file
            files_list.append(video_file)

        for i, video_file in enumerate(tqdm(files_list)):
            msg = self._check_video_size(video_file)
            df.loc[len(df)] = [video_file, msg]
            if i % 299 == 0: self.save_report()

    def check_dectime(self):
        df = self.error_df
        files_list = []
        for _ in self._iterate(deep=5):
            dectime_log = self.state.dectime_log
            files_list.append(dectime_log)

        for i, dectime_log in enumerate(tqdm(files_list)):
            msg = self._verify_encode_log(dectime_log)
            df.loc[len(df)] = [dectime_log, msg]
            if i % 299 == 0: self.save_report()

    def save_report(self):
        folder = f"{self.state.project}/check_dectime"
        os.makedirs(folder, exist_ok=True)

        filename = f'{folder}/{self.role.name}.log'
        self.error_df.to_csv(filename)

        pretty_json = json.dumps(Counter(self.error_df['msg']), indent=2)
        print(f'RESUMO: {self.role.name}\n'
              f'{pretty_json}')

        filename = f'{folder}/{self.role.name}-resume.log'
        with open(filename, 'w') as f:
            json.dump(Counter(self.error_df['msg']), f, indent=2)

    def _check_video_size(self, video_file, check_gop=False) -> str:
        size = self.check_file_size(video_file)

        if size > 0:
            if check_gop:
                gop_len = self.check_video_gop(video_file)[0]
                if not gop_len == self.config.gop:
                    return f'wrong_gop_size_{gop_len}'
            return 'apparently_ok'
        elif size == 0:
            self._clean(video_file)
            return 'filesize==0'
        elif size < 0:
            self._clean(video_file)
            return 'video_not_found'

    def _verify_encode_log(self, video_file) -> str:
        log_file = self.get_logfile(video_file)

        if not os.path.isfile(log_file):
            return 'logfile_not_found'

        with open(log_file, 'r', encoding='utf-8') as f:
            ok = bool(['' for line in f if 'Global PSNR' in line])

        if not ok:
            self._clean(video_file)
            return 'log_corrupt'

        return 'apparently_ok'

    def _verify_dectime_log(self, dectime_log) -> str:
        count_decode = self.count_decoding(dectime_log)
        if count_decode == -1:
            self._clean(dectime_log)
            return f'log_corrupt'
        return f'decoded_{count_decode}x'

    def _clean(self, video_file):
        if self.rem_error:
            self.rem_file(video_file)
            log = CheckProject.get_logfile(video_file)
            self.rem_file(log)

    @staticmethod
    def check_video_gop(video_file) -> (int, list):
        command = f'ffprobe -show_frames "{video_file}"'

        process = run(command, shell=True, capture_output=True, encoding='utf-8', )
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

    @staticmethod
    def count_decoding(log_file: Path) -> int:
        """
        Count how many times the word "utime" appears in "log_file"
        :param log_file: A path-to-file string.
        :return:
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return len(['' for line in f if 'utime' in line])
        except FileNotFoundError:
            return 0
        except UnicodeDecodeError:
            return -1

    @staticmethod
    def rem_file(file) -> None:
        if os.path.isfile(file):
            os.remove(file)

    @staticmethod
    def check_file_size(video_file) -> int:
        if not os.path.isfile(video_file):
            return -1
        filesize = os.path.getsize(video_file)
        if filesize == 0:
            return 0
        return filesize

    @staticmethod
    def menu(options_txt: list) -> int:
        options, menu = CheckProject.make_menu(options_txt)

        c = None
        while c not in options:
            c = input(menu)

        return int(c)

    @staticmethod
    def make_menu(options_txt: list) -> (list, str):
        options = [str(o) for o in range(len(options_txt))]
        menu = ['Options:']
        menu.extend([f'{o} - {text}'
                     for o, text in zip(options, options_txt)])
        menu.append(':')
        menu = '\n'.join(menu)
        return options, menu
