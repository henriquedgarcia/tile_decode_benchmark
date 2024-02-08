import datetime
from collections import defaultdict
from logging import warning
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .assets import GlobalPaths, Config, Log, AutoDict, Bcolors, Utils, SiTi
from .util import splitx, save_json, load_json, show, run_command, decode_file, get_times


class TileDecodeBenchmarkPaths(GlobalPaths, Utils, Log):
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
    def basename2(self):
        return Path(f'{self.name}_{self.resolution}_{self.fps}/'
                    f'{self.tiling}/'
                    f'{self.config["rate_control"]}{self.quality}/')

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
        folder = self.project_path / self.compressed_folder / self.basename2
        folder.absolute().mkdir(parents=True, exist_ok=True)
        return folder / f'tile{self.tile}.mp4'

    @property
    def compressed_log(self) -> Path:
        compressed_log = self.compressed_file.with_suffix('.log')
        return compressed_log

    @property
    def segments_folder(self) -> Path:
        folder = self.project_path / self.segment_folder / self.basename2 / f'tile{self.tile}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def segment_file(self) -> Path:
        chunk = int(str(self.chunk))
        return self.segments_folder / f'tile{self.tile}_{chunk:03d}.mp4'

    @property
    def segment_log(self) -> Path:
        return self.segments_folder / f'tile{self.tile}.log'

    @property
    def segment_reference_log(self) -> Path:
        qlt = self.quality
        self.quality = '0'
        segment_log = self.segment_log
        self.quality = qlt
        return segment_log

    @property
    def reference_segment(self) -> Union[Path, None]:
        qlt = self.quality
        self.quality = '0'
        segment_file = self.segment_file
        self.quality = qlt
        return segment_file

    @property
    def reference_compressed(self) -> Union[Path, None]:
        qlt = self.quality
        self.quality = '0'
        compressed_file = self.compressed_file
        self.quality = qlt
        return compressed_file

    @property
    def _dectime_folder(self) -> Path:
        folder = self.project_path / self.dectime_folder / self.basename2 / f'tile{self.tile}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def dectime_log(self) -> Path:
        chunk = int(str(self.chunk))
        return self._dectime_folder / f'chunk{chunk:03d}.log'

    @property
    def dectime_result_json(self) -> Path:
        """
        By Video
        :return:
        """
        folder = self.project_path / self.dectime_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'time_{self.video}.json'

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

    def __init__(self, config: str):
        self.config = Config(config)
        self.start_log()
        self.print_resume()
        with self.logger():
            self.main()

    def main(self): ...


class Prepare(TileDecodeBenchmarkPaths):
    def main(self):
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
    def main(self):
        for self.video in self.videos_list:  # if self.video != 'chariot_race_erp_nas': continue
            for self.tiling in self.tiling_list:
                with self.multi() as _:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            self.worker()

    def clean_compress(self):
        self.compressed_log.unlink(missing_ok=True)
        self.compressed_file.unlink(missing_ok=True)

    def skip(self, decode=False):
        # first Lossles file
        if not self.lossless_file.exists():
            self.log(f'The lossless_file not exist.', self.lossless_file)
            print(f'The file {self.lossless_file} not exist. Skipping.')
            return True

        # second check compressed
        try:
            compressed_file_size = self.compressed_file.stat().st_size
        except FileNotFoundError:
            compressed_file_size = 0

        # third Check Logfile
        try:
            compressed_log_text = self.compressed_log.read_text()
        except FileNotFoundError:
            compressed_log_text = ''

        if compressed_file_size == 0 or compressed_log_text == '':
            self.clean_compress()
            return False

        if 'encoded 1800 frames' not in compressed_log_text:
            self.log('compressed_log is corrupt', self.compressed_log)
            print(f'{Bcolors.FAIL}The file {self.compressed_log} is corrupt. Skipping.{Bcolors.ENDC}')
            self.clean_compress()
            return False

        if 'encoder         : Lavc59.18.100 libx265' not in compressed_log_text:
            self.log('CODEC ERROR', self.compressed_log)
            print(f'{Bcolors.FAIL}The file {self.compressed_log} have codec different of Lavc59.18.100 libx265. Skipping.{Bcolors.ENDC}')
            self.clean_compress()
            return False

        # decodifique os comprimidos
        if decode:
            stdout = decode_file(self.compressed_file)
            if "frame= 1800" not in stdout:
                print(f'{Bcolors.FAIL}Compress Decode Error. Cleaning.{Bcolors.ENDC}.')
                self.log(f'Compress Decode Error.', self.compressed_file)
                self.clean_compress()
                return False

        print(f'{Bcolors.FAIL}The file {self.compressed_file} is OK.{Bcolors.ENDC}')
        return True

    def worker(self):
        if self.skip():
            return

        print(f'==== Processing {self.compressed_file} ====')
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
        if factor == 'qp':
            cmd[-1] += ':ipratio=1:pbratio=1'
        cmd += [f'-vf crop='
                f'w={x2 - x1}:h={y2 - y1}:'
                f'x={x1}:y={y1}']
        cmd += [f'{self.compressed_file.as_posix()}']
        cmd = ' '.join(cmd)

        cmd = f'bash -c "{cmd}&> {self.compressed_log.as_posix()}"'
        self.command_pool.append(cmd)


class Segment(TileDecodeBenchmarkPaths):
    def main(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                with self.multi() as _:
                    for self.quality in self.quality_list:
                        for self.tile in self.tile_list:
                            self.worker()

    def worker(self) -> Any:
        if self.skip():
            return

        print(f'==== Segment {self.compressed_file} ====')
        # todo: Alternative:
        # ffmpeg -hide_banner -i {compressed_file} -c copy -f segment -segment_t
        # ime 1 -reset_timestamps 1 output%03d.mp4

        cmd = f'bash -k -c '
        cmd += '"bin/MP4Box '
        cmd += '-split 1 '
        cmd += f'{self.compressed_file.as_posix()} '
        cmd += f"-out {self.segments_folder.as_posix()}/tile{self.tile}_'$'num%03d$.mp4"
        cmd += f'|& tee {self.segment_log.as_posix()}"'

        self.command_pool.append(cmd)
        # run_command(cmd)

    def skip(self, decode=False):
        # first compressed file
        if not self.compressed_file.exists():
            self.log('compressed_file NOTFOUND.', self.compressed_file)
            print(f'{Bcolors.FAIL}The file {self.compressed_file} not exist. Skipping.{Bcolors.ENDC}')
            return True

        # second check segment log
        try:
            segment_log = self.segment_log.read_text()
        except FileNotFoundError:
            print(f'{Bcolors.FAIL}Segmentlog no exist. Cleaning.{Bcolors.ENDC}')
            self.log(f'Segmentlog no exist. The file {self.segment_log} exist.', self.segment_log)
            self.clean_segments()
            return False

        if 'file 60 done' not in segment_log:
            # Se log tem um problema, exclua o log, seus possiveis segmentos.
            # self.compressed_file.unlink(missing_ok=True)
            # self.compressed_log.unlink(missing_ok=True)
            print(f'{Bcolors.FAIL}The file {self.segment_log} is corrupt. Processing.{Bcolors.ENDC}')
            self.log('Segment_log is corrupt. Cleaning', self.segment_log)
            self.clean_segments()
            return False

        # Se log está ok; verifique os segmentos.
        for self.chunk in self.chunk_list:
            # segmento existe
            try:
                segment_file_size = self.segment_file.stat().st_size
            except FileNotFoundError:
                # um segmento não existe e o Log diz que está ok. limpeza.
                print(f'{Bcolors.FAIL}Segmentlog is OK. The file not exist. Cleaning.{Bcolors.ENDC}')
                self.log(f'Segmentlog is OK. The file {self.segment_file} not exist.', self.segment_log)
                self.clean_segments()
                return False

            if segment_file_size == 0:
                # um segmento size 0 e o Log diz que está ok. limpeza.
                print(f'{Bcolors.FAIL}Segmentlog is OK. The file SIZE 0. Cleaning.{Bcolors.ENDC}')
                self.log(f'Segmentlog is OK. The file {self.segment_file} SIZE 0', self.segment_file)
                self.clean_segments()
                return False

            # decodifique os segmentos
            if decode:
                stdout = decode_file(self.segment_file)

                if "frame=   30" not in stdout:
                    print(f'{Bcolors.FAIL}Segment Decode Error. Cleaning.{Bcolors.ENDC}.')
                    self.log(f'Segment Decode Error.', self.segment_file)
                    self.clean_segments()
                    return False

        print(f'{Bcolors.FAIL}The {self.segment_log} IS OK. Skipping.{Bcolors.ENDC}')
        return True

    def clean_segments(self):
        self.segment_log.unlink(missing_ok=True)
        for self.chunk in self.chunk_list:
            self.segment_file.unlink(missing_ok=True)


class Decode(TileDecodeBenchmarkPaths):
    turn: int

    def main(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for _ in range(5):
                        for self.tile in self.tile_list:
                            for self.chunk in self.chunk_list:
                                self.worker()

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass

        return quality_list

    def clean_dectime_log(self):
        self.dectime_log.unlink(missing_ok=True)

    def skip(self):
        self.turn = 0
        try:
            content = self.dectime_log.read_text(encoding='utf-8').splitlines()
            times = get_times(content)
            self.turn = len(times)
            if self.turn < self.config['decoding_num']:
                raise FileNotFoundError
            print(f'The segment is decoded enough.')
            return True
        except FileNotFoundError:
            if self.segment_file.exists():
                return False
            else:
                print(f'{Bcolors.WARNING}  The segment not exist. Skipping.'
                      f'{Bcolors.ENDC}')
                self.log("segment_file not exist.", self.segment_file)
                return True

    def worker(self) -> Any:
        if self.skip():
            return

        print(f'Decoding file "{self.segment_file}". Turn {self.turn + 1}')
        stdout = decode_file(self.segment_file, threads=1)
        with self.dectime_log.open('a') as f:
            f.write(f'\n==========\n{stdout}')


class GetBitrate(TileDecodeBenchmarkPaths):
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
    turn: int
    _video: str
    skip_time: bool
    skip_rate: bool
    result_rate: AutoDict
    result_times: AutoDict
    change_flag: bool

    def main(self):
        for self.video in self.videos_list:
            self.result_rate = AutoDict()
            if self.skip1(): continue

            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for self.tile in self.tile_list:
                        for self.chunk in self.chunk_list:
                            self.bitrate()

            if self.change_flag:
                save_json(self.result_rate, self.bitrate_result_json)

    def skip1(self, check_result=True):
        self.change_flag = False
        if self.bitrate_result_json.exists():
            print(f'\n[{self.vid_proj}][{self.video}] - The result_json exist.')
            if check_result:
                self.result_rate = load_json(self.bitrate_result_json,
                                             object_hook=AutoDict)
                return False
            return True

    def bitrate(self) -> Any:
        print(f'\r[{self.vid_proj}][{self.video}][{self.tiling}][CRF{self.quality}][tile{self.tile}][chunk{self.chunk}]]', end='')

        try:
            chunk_size = self.segment_file.stat().st_size
            if chunk_size == 0:
                self.log('BITRATE==0', self.segment_file)
                return
        except FileNotFoundError:
            self.log('SEGMENT_FILE_NOT_FOUND', self.segment_file)
            return

        bitrate = 8 * chunk_size / self.chunk_dur

        old_bitrate = self.result_rate[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
        if bitrate == old_bitrate:
            return
        if not self.change_flag: self.change_flag = True
        self.result_rate[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = bitrate

    @property
    def quality_list(self) -> list[str]:
        quality_list: list = self.config['quality_list']

        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


class GetDectime(TileDecodeBenchmarkPaths):
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
    turn: int
    _video: str
    skip_time: bool
    skip_rate: bool
    result_rate: AutoDict
    result_times: AutoDict

    def __init__(self, config: str):
        self.log_text = defaultdict(list)
        self.config = Config(config)
        for self.video in self.videos_list:
            self.get_dectime()
        path = Path(f'LogResults_{datetime.datetime.now()}.csv'.replace(':', '-'))
        pd.DataFrame(self.log_text).to_csv(path, encoding='utf-8')

    def get_dectime(self):
        if self.dectime_result_json.exists():
            print(f'\nThe file {self.dectime_result_json} exist and not '
                  f'overwrite. Skipping.')
            return

        self.result_times = AutoDict()
        for self.tiling in self.tiling_list:
            for self.quality in self.quality_list:
                for self.tile in self.tile_list:
                    for self.chunk in self.chunk_list:
                        self.dectime()

        save_json(self.result_times, self.dectime_result_json)

    def dectime(self) -> Any:
        print(f'\rDectime [{self.vid_proj}][{self.name}][{self.tiling}][crf{self.quality}][tile{self.tile}]'
              f'[chunk{self.chunk}] = ', end='')
        try:
            content = self.dectime_log.read_text(encoding='utf-8').splitlines()
            times = self.get_times(content)
        except FileNotFoundError:
            print(f'\n{Bcolors.FAIL}    The dectime log not exist. Skipping.'
                  f'{Bcolors.ENDC}')
            self.result_times[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = [10, 10, 10]
            self.log('DECTIME_FILE_NOT_FOUND', self.dectime_log)
            return
        try:
            times = sorted(times)[-3:]
        except TypeError:
            print(f'{Bcolors.WARNING} The times is not a list. {type(times)}.{Bcolors.ENDC}')
            self.log('times is not a list', self.dectime_log)
            return

        if len(times) < self.config['decoding_num']:
            print(f'\n{Bcolors.WARNING}    The dectime is lower than 3: {times}.{Bcolors.ENDC}')
            self.log(f'DECTIME_NOT_DECODED_ENOUGH_{len(times)}', self.dectime_log)

        if 0 in times:
            print(f'\n{Bcolors.WARNING}    0  found in {times}{Bcolors.ENDC}')
            self.log('DECTIME_ZERO_FOUND', self.dectime_log)
        else:
            print(f' {times}', end='')
        self.result_times[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk] = times


class MakeSiti(TileDecodeBenchmarkPaths):
    def main(self):
        self._make_siti()
        self._scatter_plot_siti()
        pd.DataFrame(self.log_text).to_csv(Path(f'LogTestResultsDectime_{datetime.datetime.now()}.csv'.replace(':', '-')), encoding='utf-8')

    def _make_siti(self):
        for self.video in self.videos_list:
            if self.siti_results.exists():
                continue

            self.tiling = '1x1'
            self.quality = '28'
            self.tile = '0'

            if not self.compressed_file.exists():
                print(f'File not exist {self.compressed_file}')
                continue

            siti = SiTi(self.compressed_file)
            siti_results_df = {f'{self.video}': siti.siti}
            save_json(siti_results_df, self.siti_results)

    def _scatter_plot_siti(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=300)
        fig: plt.Figure
        fig2, ax3 = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

        for self.video in self.videos_list:
            siti_results_df = load_json(self.siti_results)
            si = siti_results_df[f'{self.video}']['si']
            ti = [0] + siti_results_df[f'{self.video}']['ti']
            si_med = np.median(si)
            ti_med = np.median(ti)
            ax1.plot(si, label=self.name.replace('_nas', ''))
            ax2.plot(ti, label=self.name.replace('_nas', ''))
            ax3.scatter(si_med, ti_med, label=self.name.replace('_nas', ''))

        # ax.set_title('Si/Ti', fontdict={'size': 16})
        ax3.set_xlabel("Spatial Information")
        ax3.set_ylabel("Temporal Information")
        ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                   fontsize='small')

        fig2.suptitle('SI x TI')
        fig2.tight_layout()
        fig2.savefig(self.project_path / self._siti_folder / 'scatter.png')
        show(fig2)

        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Spatial Information")
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Temporal Information')
        handles, labels = ax1.get_legend_handles_labels()
        fig.suptitle('SI/TI by frame')
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=[0.8, 0.93],
                   fontsize='small')
        fig.tight_layout()
        fig.subplots_adjust(right=0.78)
        fig.savefig(self.project_path / self._siti_folder / 'plot.png')
        fig.show()


TileDecodeBenchmarkOptions = {'0': Prepare,  # 0
                              '1': Compress,  # 1
                              '2': Segment,  # 2
                              '3': Decode,  # 3
                              '4': GetBitrate,  # 4
                              '5': GetDectime,  # 5
                              '6': MakeSiti,  # 6
                              }
