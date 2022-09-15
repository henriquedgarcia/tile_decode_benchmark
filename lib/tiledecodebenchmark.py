from enum import Enum
from logging import warning
from pathlib import Path
from typing import Any, Union

from .assets2 import Base, GlobalPaths
from .util2 import splitx, run_command, AutoDict, save_json


class TileDecodeBenchmarkPaths(GlobalPaths):
    # Folders
    original_folder = Path('original')
    lossless_folder = Path('lossless')
    compressed_folder = Path('compressed')
    segment_folder = Path('segment')
    _viewport_folder = Path('viewport')
    _siti_folder = Path('siti')
    _check_folder = Path('check')

    @property
    def basename(self):
        return Path(f'{self.name}_'
                    f'{self.resolution}_'
                    f'{self.fps}_'
                    f'{self.tiling}_'
                    f'{self.config["rate_control"]}{self.quality}')

    @property
    def original_file(self) -> Path:
        return self.original_folder / self.original

    @property
    def lossless_file(self) -> Path:
        folder = self.project_path / self.lossless_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video}_{self.resolution}_{self.config["fps"]}.mp4'

    @property
    def compressed_file(self) -> Path:
        folder = self.project_path / self.compressed_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def segment_file(self) -> Path:
        folder = self.project_path / self.segment_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.mp4'

    @property
    def reference_segment(self) -> Union[Path, None]:
        # 'segment/angel_falls_nas_4320x2160_30_12x8_crf0/tile11_030.mp4'
        basename = Path(f'{self.name}_'
                        f'{self.resolution}_'
                        f'{self.fps}_'
                        f'{self.tiling}_'
                        f'{self.config["rate_control"]}{self.config["original_quality"]}')
        folder = self.project_path / self.segment_folder / basename
        return folder / f'tile{self.tile}_{int(self.chunk):03d}.mp4'

    @property
    def reference_compressed(self) -> Union[Path, None]:
        # 'compressed/angel_falls_nas_4320x2160_30_12x8_crf0/tile11.mp4'
        basename = Path(f'{self.name}_'
                        f'{self.resolution}_'
                        f'{self.fps}_'
                        f'{self.tiling}_'
                        f'{self.config["rate_control"]}{self.config["original_quality"]}')
        folder = self.project_path / self.compressed_folder / basename
        return folder / f'tile{self.tile}.mp4'

    @property
    def dectime_log(self) -> Path:
        folder = self.project_path / self.dectime_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        chunk = int(str(self.chunk))
        return folder / f'tile{self.tile}_{chunk:03d}.log'

    @property
    def dectime_result_json(self) -> Path:
        """
        By Video
        :return:
        """
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'times_{self.video}.json'

    @property
    def bitrate_result_json(self) -> Path:
        folder = self.project_path / self.segment_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'rate_{self.video}.json'


class Prepare(TileDecodeBenchmarkPaths):
    def loop(self):
        for self.video in self.videos_list:
            yield

    def worker(self, overwrite=False):
        original_file: Path = self.original_file
        lossless_file: Path = self.lossless_file

        if lossless_file and not overwrite:
            warning(f'  The file {lossless_file=} exist. Skipping.')
            return

        if not original_file.exists():
            warning(f'  The file {original_file=} not exist. Skipping.')
            return

        resolution_ = splitx(self.resolution)
        dar = resolution_[0] / resolution_[1]

        cmd = f'ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {self.offset} '
        cmd += f'-i {original_file} '
        cmd += f'-crf 0 '
        cmd += f'-t {self.duration} '
        cmd += f'-r {self.fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf "scale={self.resolution},setdar={dar}" '
        cmd += f'{lossless_file}'

        print(cmd)
        lossless_log: Path = self.lossless_file.with_suffix('.log')
        run_command(cmd, lossless_log, 'w')


class Compress(TileDecodeBenchmarkPaths):
    def loop(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        print(f'\r{self.compressed_file}', end='')
                        yield

    def worker(self, overwrite=False):
        if self.compressed_file.exists() and not overwrite:
            warning(f'The file {self.compressed_file} exist. Skipping.')
            return

        if not self.lossless_file.exists():
            warning(f'The file {self.lossless_file} not exist. Skipping.')
            return

        pw, ph = splitx(self.resolution)
        M, N = splitx(self.tiling)
        tw, th = int(pw / M), int(ph / N)
        tx, ty = int(self.tile) * tw, int(self.tile) * th
        factor = self.config["rate_control"]

        cmd = ['bin/ffmpeg -hide_banner -y -psnr']
        cmd += [f'-i {self.lossless_file}']
        cmd += [f'-c:v libx265']
        cmd += [f'-{factor} {self.quality} -tune "psnr"']
        cmd += [f'-x265-params']
        cmd += [f'"keyint={self.gop}:'
                f'min-keyint={self.gop}:'
                f'open-gop=0:'
                f'scenecut=0:'
                f'info=0"']
        cmd += [f'-vf "crop='
                f'w={tw}:h={th}:'
                f'x={tx}:y={ty}"']
        cmd += [f'{self.compressed_file}']
        cmd = ' '.join(cmd)
        compressed_log = self.compressed_file.with_suffix('.log')
        run_command(cmd, compressed_log, 'w')


class Segment(TileDecodeBenchmarkPaths):
    def loop(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tiling_list:
                        print(f'==== Processing {self.compressed_file} ====')
                        yield

    def worker(self, overwrite=False) -> Any:
        segment_log = self.segment_file.with_suffix('.log')

        # If segment log size is very small, infers error and overwrite.
        if segment_log.is_file() and segment_log.stat().st_size > 10000 and not overwrite:
            warning(f'The file {segment_log} exist. Skipping.')
            return

        if not self.compressed_file.is_file():
            warning(f'The file {self.compressed_file} not exist. Skipping.')
            return

        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4

        cmd = ['MP4Box']
        cmd += ['-split 1']
        cmd += [f'{self.compressed_file}']
        cmd += [f'-out {self.segment_folder}/']
        cmd = ' '.join(cmd)
        cmd = f'bash -c "{cmd}"'

        run_command(cmd, segment_log, 'w')


class Decode(TileDecodeBenchmarkPaths):
    def loop(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for turn in range(self.decoding_num):
                        for self.tile in self.tiling_list:
                            for self.chunk in self.chunk_list:
                                print(f'Decoding {self.segment_file=}. {turn = }', end='')
                                yield

    def worker(self, overwrite=False) -> Any:
        if self.dectime_log.exists():
            if self.decoding_num - self.count_decoding(self.dectime_log) <= 0 and not overwrite:
                warning(f'  {self.segment_file} is decoded enough. Skipping.')
                return

        if not self.segment_file.is_file():
            warning(f'  The file {self.segment_file} not exist. Skipping.')
            return

        cmd = (f'ffmpeg -hide_banner -benchmark '
               f'-codec hevc -threads 1 '
               f'-i {self.segment_file} '
               f'-f null -')

        run_command(cmd, self.dectime_log, 'a')


class Result(TileDecodeBenchmarkPaths):
    """
       The result dict have a following structure:
       results[video_name][tile_pattern][quality][tile_id][chunk_id]
               ['times'|'rate']
       [video_proj]    : The video projection
       [video_name]    : The video name
       [tile_pattern]  : The tile tiling. e.g. "6x4"
       [quality]       : Quality. An int like in crf or qp.
       [tile_id]           : the tile number. ex. max = 6*4
       [chunk_id]           : the chunk number. Start with 1.

       'times': list(float, float, float)
       'rate': float
       """

    skip_time: bool
    skip_rate: bool
    result_rate: AutoDict
    result_times: AutoDict

    def loop(self, overwrite=False):
        for self.video in self.videos_list:
            if self.dectime_result_json.exists() and not overwrite:
                self.skip_time = True
                print(f'The file {self.dectime_result_json} exist and not overwrite. Skipping.')
            if self.bitrate_result_json.exists() and not overwrite:
                self.skip_rate = True
                print(f'The file {self.dectime_result_json} exist and not overwrite. Skipping.')

            self.result_rate = AutoDict()
            self.result_times = AutoDict()

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tiling_list:
                        for self.chunk in self.chunk_list:
                            print(f'Collecting {self.segment_file}')
                            yield

            self.save_dectime()

    def worker(self) -> Any:
        if not self.skip_rate:
            result_rate = self.result_rate[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            if self.segment_file.exists():
                try:
                    chunk_size = self.segment_file.stat().st_size
                    bitrate = 8 * chunk_size / self.chunk_dur
                    result_rate['rate'] = bitrate
                except PermissionError:
                    warning(f'PermissionError error on reading size of {self.segment_file}.')
                    result_rate['rate'] = 0
            else:
                warning(f'The chunk {self.segment_file} not exist_ok.')
                result_rate['rate'] = 0

        if not self.skip_time:
            results_times = self.result_times[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]

            if self.dectime_log.exists():
                content = self.dectime_log.read_text(encoding='utf-8').splitlines()
                times = [float(line.strip().split(' ')[1].split('=')[1][:-1])
                         for line in content if 'utime' in line]
                results_times['times'] = times
            else:
                warning(f'The dectime log {self.dectime_log} not exist. Skipping.')
                results_times['times'] = [6, 6, 6]

    def save_dectime(self):
        filename = self.dectime_result_json
        save_json(self.result_times, filename)

        filename = self.bitrate_result_json
        save_json(self.result_rate, filename)


class TileDecodeBenchmarkOptions(Enum):
    PREPARE = 0
    COMPRESS = 1
    SEGMENT = 2
    DECODE = 3
    COLLECT_RESULTS = 4

    def __repr__(self):
        return str({self.value: self.name})


class TileDecodeBenchmark(Base):
    operations = {'PREPARE': Prepare,
                  'COMPRESS': Compress,
                  'SEGMENT': Segment,
                  'DECODE': Decode,
                  'COLLECT_RESULTS': Result}