import dectime.util as util
from os import path
import subprocess
from enum import Enum



class Role(Enum):
    PREPARE = 0
    COMPRESS = 1
    SEGMENT = 2
    DECODE = 3
    RESULTS = 4


class Dectime:
    def __init__(self, config: str):
        self.state = util.VideoState(config)
        self.results = util.AutoDict()

    @staticmethod
    def _check_existence(file_path, overwrite):
        if path.isfile(f'{file_path}') and not overwrite:
            msg = 'Skipping' if overwrite else 'Overwriting'
            print(f'The file "{file_path}" exist. {msg}')
            return True

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

        log = f'{uncompressed_file[:-4]}.txt'
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
        frame = self.state.frame
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

        log = f'{compressed_file[:-4]}.txt'
        with open(log, 'a', encoding='utf-8') as f:
            subprocess.run(command, shell=True, stderr=subprocess.STDOUT,
                           stdout=f)

    def segment(self, overwrite=False):
        log = f'{self.state.segment_folder}/tile{self.state.tile.id}.txt'
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

    def collect_result(self) -> util.AutoDict:
        # Collect decode time
        self.results[
            self.state.video.name][
            self.state.pattern.pattern][
            self.state.quality][
            self.state.tile.id][
            self.state.chunk] = self._collect_dectime()

        # Collect quality
        if self.state.chunk == 1:
            self.results[
                self.state.video.name][
                self.state.pattern.pattern][
                self.state.quality][
                self.state.tile.id] = self._collect_psnr()
        return self.results

    def _collect_dectime(self) -> util.AutoDict:
        """

        :param self:
        :return:
        """
        dectime = util.AutoDict()
        dectime_file = self.state.dectime_file
        segment_file = self.state.segment_file

        get_time = lambda line: float(line.strip().split('=')[1][:-1])
        with open(dectime_file, 'r', encoding='utf-8') as f:
            dectime['utime'] = [get_time(line) for line in f
                                if 'bench:' in line]

        chunk_size = path.getsize(segment_file)
        dectime['bitrate'] = chunk_size * 8 / (self.state.gop / self.state.fps)

        return dectime

    def _collect_psnr(self):
        psnr = util.AutoDict('list')

        get_psnr = lambda l: float(l.strip().split(',')[3].split(':')[1])
        get_qp = lambda l: float(l.strip().split(',')[2].split(':')[1])

        with open(f'{self.state.compressed_file:-4}.txt', 'r',
                  encoding='utf-8') as f:
            for line in f:
                if 'Global PSNR' in line:
                    psnr['psnr'] = get_psnr(line)
                    psnr['qp_avg'] = get_qp(line)
                    break
        return psnr

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
                                self.decode()
                                continue
