from enum import Enum
from logging import warning, error
from pathlib import Path
from typing import Any, Union
from subprocess import run, STDOUT

from matplotlib import pyplot as plt

from .assets2 import Base, GlobalPaths
from .siti import SiTi
from .util2 import splitx, AutoDict, save_json, load_json


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
        return folder / f'{self.name}_{self.resolution}_{self.config["fps"]}.mp4'

    @property
    def compressed_file(self) -> Path:
        folder = self.project_path / self.compressed_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def segments_folder(self) -> Path:
        folder = self.project_path / self.segment_folder / self.basename
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def segment_file(self) -> Path:
        chunk = int(str(self.chunk))
        return self.segments_folder / f'tile{self.tile}_{chunk:03d}.mp4'

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

    @property
    def siti_results(self) -> Path:
        folder = self.project_path / self._siti_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'{self.video}_siti_results.json'


class Prepare(TileDecodeBenchmarkPaths):
    def __init__(self):
        for self.video in self.videos_list:
            self.worker()

    def worker(self, overwrite=False):
        original_file: Path = self.original_file
        lossless_file: Path = self.lossless_file
        lossless_log: Path = self.lossless_file.with_suffix('.log')

        if lossless_file and not overwrite:
            warning(f'  The file {lossless_file=} exist. Skipping.')
            return

        if not original_file.exists():
            warning(f'  The file {original_file=} not exist. Skipping.')
            return

        resolution_ = splitx(self.resolution)
        dar = resolution_[0] / resolution_[1]

        cmd = f'bin/ffmpeg '
        cmd += f'-hide_banner -y '
        cmd += f'-ss {self.offset} '
        cmd += f'-i {original_file.as_posix()} '
        cmd += f'-crf 0 '
        cmd += f'-t {self.duration} '
        cmd += f'-r {self.fps} '
        cmd += f'-map 0:v '
        cmd += f'-vf scale={self.resolution},setdar={dar} '
        cmd += f'{lossless_file.as_posix()}'

        cmd = f'bash -c "{cmd}|& tee {lossless_log.as_posix()}"'
        print(cmd)
        run_command(cmd)


class Compress(TileDecodeBenchmarkPaths):
    def __init__(self):
        self.print_resume()
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        print(f'==== Processing {self.compressed_file} ====')
                        self.worker()

    def worker(self, overwrite=False):
        if self.compressed_file.exists() and not overwrite:
            warning(f'The file {self.compressed_file} exist. Skipping.')
            return

        if not self.lossless_file.exists():
            warning(f'The file {self.lossless_file} not exist. Skipping.')
            return

        x1, y1, x2, y2 = self.tile_position()

        factor = self.config["rate_control"]

        cmd = ['bin/ffmpeg -hide_banner -y -psnr']
        cmd += [f'-i {self.lossless_file.as_posix()}']
        cmd += [f'-c:v libx265']
        cmd += [f'-{factor} {self.quality} -tune psnr']
        cmd += [f'-x265-params']
        cmd += [f'keyint={self.gop}:'
                f'min-keyint={self.gop}:'
                f'open-gop=0:'
                f'scenecut=0:'
                f'info=0']
        cmd += [f'-vf crop='
                f'w={x2-x1}:h={y2-y1}:'
                f'x={x1}:y={y1}']
        cmd += [f'{self.compressed_file.as_posix()}']
        cmd = ' '.join(cmd)

        compressed_log = self.compressed_file.with_suffix('.log')
        cmd = f'bash -c "{cmd}|& tee {compressed_log.as_posix()}"'
        run_command(cmd)


class Segment(TileDecodeBenchmarkPaths):
    def __init__(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        print(f'==== Processing {self.compressed_file} ====')
                        self.worker()

    def worker(self, overwrite=False) -> Any:
        segment_log = self.segments_folder / f'tile{self.tile}.log'

        # If segment log size is very small, infers error and overwrite.
        if segment_log.is_file() and segment_log.stat().st_size > 50000 and not overwrite:
            warning(f'The file {segment_log} exist. Skipping.')
            return

        if not self.compressed_file.is_file():
            warning(f'The file {self.compressed_file} not exist. Skipping.')
            return

        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4

        cmd = ['bin/MP4Box']
        cmd += ['-split 1']
        cmd += [f'{self.compressed_file.as_posix()}']
        cmd += [f'-out {self.segments_folder.as_posix()}/tile{self.tile}_\$num%03d$.mp4']
        cmd = ' '.join(cmd)
        cmd = f"bash -c '{cmd} |&tee {segment_log.as_posix()}'"

        run_command(cmd)


class Decode(TileDecodeBenchmarkPaths):
    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']
        
        try:
            quality_list.remove('0')
        except ValueError:
            None
            
        return quality_list

    def __init__(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for turn in range(self.decoding_num):
                            for self.chunk in self.chunk_list:
                                print(f'Decoding {self.segment_file=}. {turn = }', end='')
                                self.worker()

    def worker(self, overwrite=False) -> Any:
        if self.dectime_log.exists():
            if self.decoding_num - self.count_decoding(self.dectime_log) <= 0 and not overwrite:
                warning(f'  {self.segment_file} is decoded enough. Skipping.')
                return

        if not self.segment_file.is_file():
            warning(f'  The file {self.segment_file} not exist. Skipping.')
            return

        cmd = (f'bin/ffmpeg -hide_banner -benchmark '
               f'-codec hevc -threads 1 '
               f'-i {self.segment_file.as_posix()} '
               f'-f null -')
        cmd = f'bash -c "{cmd}|& tee {self.dectime_log.as_posix()}"'

        run_command(cmd)


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


class MakeSiti(TileDecodeBenchmarkPaths):
    def __init__(self):
        for self.video in self.videos_list:
            if self.siti_results.exists():
                continue

            self.tiling = '1x1'
            self.quality = '28'
            self.tile = '0'
            siti = SiTi(self.compressed_file)
            si, ti = siti.calc_siti()
            siti_results_df = {f'{self.video}_si': si,
                               f'{self.video}_ti': ti}
            save_json(siti_results_df, self.siti_results)

        self._scatter_plot_siti()

    def _scatter_plot_siti(self):
        siti_results_df = load_json(self.siti_results)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
        fig: plt.Figure
        ax: plt.Axes

        for self.video in self.videos_list:
            si = siti_results_df[f'{self.video}_si']
            ti = [0] + siti_results_df[f'{self.video}_ti']
            ax1.plot(si, label=self.name)
            ax2.plot(ti, label=self.name)

        # ax.set_title('Si/Ti', fontdict={'size': 16})
        ax1.set_xlabel("Spatial Information")
        ax2.set_xlabel('Temporal Information')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize='small')

        fig.tight_layout()
        fig.savefig(self.project_path / self._siti_folder / 'scatter.png')
        fig.show()


class TestSegments(TileDecodeBenchmarkPaths):
    def __init__(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for turn in range(self.decoding_num):
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                # print(f'Testing {self.segment_file=}.')
                                self.worker()

    def worker(self) -> Any:
        if not self.segment_file.exists():
            warning(f'  {self.segment_file} not exist..')
            return


class TileDecodeBenchmarkOptions(Enum):
    """Operation 0"""
    PREPARE = 0
    COMPRESS = 1
    SEGMENT = 2
    DECODE = 3
    COLLECT_RESULTS = 4
    SITI = 5
    TEST_SEGMENTS = 6

    def __repr__(self):
        return str({self.value: self.name})


class TileDecodeBenchmark(Base):
    operations = {'PREPARE': Prepare,
                  'COMPRESS': Compress,
                  'SEGMENT': Segment,
                  'DECODE': Decode,
                  'COLLECT_RESULTS': Result,
                  'SITI': MakeSiti,
                  'TEST_SEGMENTS': TestSegments}


def run_command(command: str):
    print(command)
    process = run(command, shell=True, stderr=STDOUT, encoding='utf-8')

    if process.returncode != 0:
        error(f'SUBPROCESS ERROR: {command=}\n'
              f'    {process.returncode = } - {process.stdout = }. Continuing.')
