import time
from abc import ABC
from collections import defaultdict, Counter
from pathlib import Path
from typing import Union

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skvideo.io import FFmpegWriter, FFmpegReader

import lib.video360 as vp
from ._tilequality import SegmentsQualityPaths
from ._tiledecodebenchmark import TileDecodeBenchmarkPaths
from .util import load_json, save_json, splitx, idx2xy
from .transform import cart2hcs, lin_interpol
from .assets import Config, AutoDict


pi = np.pi
pi2 = np.pi * 2

# "Videos 10,17,27,28 were rotated 265, 180,63,81 degrees to right,
# respectively, to reorient during playback." - Author
# Videos 'cable_cam_nas','drop_tower_nas','wingsuit_dubai_nas','drone_chases_car_nas'
# rotation = rotation_map[video_nas_id] if video_nas_id in [10, 17, 27, 28] else 0
rotation_map = {'cable_cam_nas': 265 / 180 * pi, 'drop_tower_nas': 180 / 180 * pi,
                'wingsuit_dubai_nas': 63 / 180 * pi, 'drone_chases_car_nas': 81 / 180 * pi}


class GetTilesPath(TileDecodeBenchmarkPaths, ABC):
    dataset_folder: Path
    video_id_map: dict
    user_map: dict
    _csv_dataset_file: Path
    video_name: str
    user_id: str
    head_movement: pd.DataFrame

    # @property
    # def workfolder(self) -> Path:
    #     """
    #     Need None
    #     :return:
    #     """
    #     if self._workfolder is None:
    #         folder = self.get_tiles_path / f'{self.__class__.__name__}'
    #         folder.mkdir(parents=True, exist_ok=True)
    #         return folder
    #     else:
    #         return self._workfolder
    #
    # @workfolder.setter
    # def workfolder(self, value):
    #     self._workfolder = value

    # <editor-fold desc="Dataset Path">
    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def dataset_folder(self) -> Path:
        return Path('datasets') / self.dataset_name

    @property
    def dataset_json(self) -> Path:
        return self.dataset_folder / f'{self.config["dataset_name"]}.json'

    @property
    def csv_dataset_file(self) -> Path:
        return self._csv_dataset_file

    @csv_dataset_file.setter
    def csv_dataset_file(self, value):
        self._csv_dataset_file = value
        user_nas_id, video_nas_id = self._csv_dataset_file.stem.split('_')
        self.video_name = self.video_id_map[video_nas_id]
        self.user_id = self.user_map[user_nas_id]

        names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
        self.head_movement = pd.read_csv(self.csv_dataset_file, names=names)
    # </editor-fold>


class ProcessNasrabadi(GetTilesPath):
    user_map = {}
    dataset_final = AutoDict()
    previous_line: tuple
    frame_counter: int

    def __init__(self, config):
        self.config = config
        self.print_resume()
        print(f'Processing dataset {self.dataset_folder}.')
        if self.dataset_json.exists(): return

        self.video_id_map = load_json(f'{self.dataset_folder}/videos_map.json')
        self.user_map = load_json(f'{self.dataset_folder}/usermap.json')

        for self.csv_dataset_file in self.dataset_folder.glob('*/*.csv'):
            self.frame_counter = 0
            self.worker()

        print(f'Finish. Saving as {self.dataset_json}.')
        save_json(self.dataset_final, self.dataset_json)

    def worker(self):
        # For each  csv_file
        yaw_pitch_roll_frames = []
        start_time = time.time()
        n = 0

        print(f'\rUser {self.user_id} - {self.video_name} - ', end='')
        for n, line in enumerate(self.head_movement.itertuples(index=False, name=None)):
            timestamp, Qx, Qy, Qz, Qw, Vx, Vy, Vz = map(float, line)
            xyz = np.array([Vx, -Vy, Vz])  # Based on paper

            try:
                yaw_pitch_roll = self.process_vectors((timestamp, xyz))
                yaw_pitch_roll_frames.append(list(yaw_pitch_roll))
                self.frame_counter += 1
                if self.frame_counter == 1800: break
            except ValueError:
                pass
            self.previous_line = timestamp, xyz

        yaw_pitch_roll_frames += [yaw_pitch_roll_frames[-1]] * (1800 - len(yaw_pitch_roll_frames))

        self.dataset_final[self.video_name][self.user_id] = yaw_pitch_roll_frames
        print(f'Samples {n:04d} - {self.frame_counter=} - {time.time() - start_time:0.3f} s.')

    def process_vectors(self, actual_line):
        timestamp, xyz = actual_line
        frame_timestamp = self.frame_counter / 30

        if timestamp < frame_timestamp:
            # Skip. It's not the time.
            raise ValueError
        elif timestamp > frame_timestamp:
            # Linear Interpolation
            old_timestamp, old_xyz = self.previous_line
            xyz = lin_interpol(frame_timestamp, timestamp, old_timestamp, np.array(xyz), np.array(old_xyz))

        yaw, pitch = cart2hcs(xyz).T
        roll = [0] * len(yaw) if isinstance(yaw, np.ndarray) else 0

        if self.video_name in rotation_map:
            yaw -= rotation_map[self.video_name]

        yaw = np.mod(yaw + pi, pi2) - pi
        return np.round([yaw, pitch, roll], 6).T


class GetTilesProps(GetTilesPath):
    results: AutoDict
    _dataset: dict

    @property
    def dataset(self) -> dict:
        try:
            return self._dataset
        except AttributeError:
            self._dataset = load_json(self.dataset_json)
            return self._dataset

    @property
    def users_list(self) -> list[str]:
        return list(self.dataset[self.name].keys())

    @property
    def get_tiles_folder(self) -> Path:
        folder = self.project_path / 'get_tiles'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def get_tiles_json(self) -> Path:
        path = self.get_tiles_folder / f'get_tiles_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}_fov{self.fov}.json'
        return path

    @property
    def counter_tiles_json(self):
        folder = self.get_tiles_folder / 'counter'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'counter_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}_fov{self.fov}.json'
        return path

    @property
    def heatmap_tiling(self):
        folder = self.get_tiles_folder / 'heatmap'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'heatmap_tiling_{self.dataset_name}_{self.vid_proj}_{self.name}_{self.tiling}_fov{self.fov}.png'
        return path


class GetTiles(GetTilesProps):
    projection: vp.ProjBase
    n_frames: int
    erp_list: dict[str, vp.ProjBase]
    cmp_list: dict[str, vp.ProjBase]
    tiles_1x1: dict[str, dict[str, Union[vp.ProjBase], list, dict]]

    def init(self, config):
        self.print_resume()
        self.n_frames = 1800
        self.config = Config(config)
        self.erp_list = {self.tiling: vp.ERP(self.tiling, '576x288', self.fov)
                         for self.tiling in self.tiling_list}
        self.cmp_list = {self.tiling: vp.CMP(self.tiling, '432x288', self.fov)
                         for self.tiling in self.tiling_list}
        self.tiles_1x1 = {'frame': [["0"]] * self.n_frames,
                          'chunks': {str(i): ["0"] for i in range(1, int(self.duration) + 1)}}

    def __init__(self, config):
        self.init(config)

        for self.video in self.videos_list:
            self.results = AutoDict()
            self.worker()

            # self.count_tiles()
            # self.heatmap()
            # self.plot_test()

    def worker(self):
        if self.get_tiles_json.exists(): return

        for self.tiling in self.tiling_list:
            for self.user in self.users_list:
                print(f'{self.name} - tiling {self.tiling} - User {self.user}')

                if self.tiling == '1x1':
                    self.results[self.vid_proj][self.name][self.tiling][self.user] = self.tiles_1x1
                    # self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'] = [["0"]] * self.n_frames
                    # self.results[self.vid_proj][self.name][self.tiling][self.user]['chunks'] = {str(i): ["0"] for i in range(1, int(self.duration) + 1)}
                    continue

                if self.vid_proj == 'erp':
                    self.projection = self.erp_list[self.tiling]
                elif self.vid_proj == 'cmp':
                    self.projection = self.cmp_list[self.tiling]

                result_frames = []
                result_chunks = {}
                tiles_in_chunk = set()

                for frame, (yaw_pitch_roll) in enumerate(self.dataset[self.name][self.user]):
                    vptiles: list[str] = self.projection.get_vptiles(yaw_pitch_roll)
                    tiles_in_chunk.update(vptiles)
                    result_frames.append(vptiles)

                    if (frame + 1) % 30 == 0:
                        # vptiles by chunk start from 1 gpac defined
                        chunk = frame // 30 + 1
                        result_chunks[f'{chunk}'] = list(tiles_in_chunk)
                        tiles_in_chunk.clear()

                self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'] = result_frames
                self.results[self.vid_proj][self.name][self.tiling][self.user]['chunks'] = result_chunks

        print(f'Saving {self.get_tiles_json}')
        save_json(self.results, self.get_tiles_json)

    # def diferences(self):
    #     if not self.get_tiles_json.exists():
    #         print(f'The file {self.get_tiles_json} NOT exist. Skipping')
    #         return
    #
    #     self.results1 = load_json(self.get_tiles_json)
    #     self.results2 = load_json(self.get_tiles_json.with_suffix(f'.json.old'))
    #
    #     for self.tiling in self.tiling_list:
    #         for self.user in self.users_list:
    #             keys = ('frame', 'head_positions', 'chunks',)
    #             result_frames1: list = self.results1[self.vid_proj][self.name][self.tiling][self.user]['frame']
    #             result_frames2: list = self.results2[self.vid_proj][self.name][self.tiling][self.user]['frame']
    #             result_chunks1: dict = self.results1[self.vid_proj][self.name][self.tiling][self.user]['chunks']
    #             result_chunks2: dict = self.results2[self.vid_proj][self.name][self.tiling][self.user]['chunks']
    #
    #             # head_positions1: list = self.results1[self.vid_proj][self.name][self.tiling][self.user]['head_positions']
    #             # head_positions2: list = self.results2[self.vid_proj][self.name][self.tiling][self.user]['head_positions']
    #
    #             # for get_tiles_frame1, get_tiles_frame2 in zip(result_frames1,result_frames2):
    #             print(f'[{self.name}][{self.tiling}][{self.user}] ', end='')
    #             result_frames2_str = [list(map(str, item)) for item in result_frames2]
    #             if result_frames1 == result_frames2_str:
    #                 print(f'igual')
    #             else:
    #                 print('não igual')

    def count_tiles(self):
        if self.counter_tiles_json.exists(): return

        self.results = load_json(self.get_tiles_json)
        result = {}

        for self.tiling in self.tiling_list:
            if self.tiling == '1x1': continue

            # <editor-fold desc="Count tiles">
            tiles_counter_chunks = Counter()  # Collect tiling count

            for self.user in self.users_list:
                result_chunks: dict[str, list[str]] = self.results[self.vid_proj][self.name][self.tiling][self.user]['chunks']

                for chunk in result_chunks:
                    tiles_counter_chunks = tiles_counter_chunks + Counter(result_chunks[chunk])
            # </editor-fold>

            print(tiles_counter_chunks)
            dict_tiles_counter_chunks = dict(tiles_counter_chunks)

            # <editor-fold desc="Normalize Counter">
            nb_chunks = sum(dict_tiles_counter_chunks.values())
            for self.tile in self.tile_list:
                try:
                    dict_tiles_counter_chunks[self.tile] /= nb_chunks
                except KeyError:
                    dict_tiles_counter_chunks[self.tile] = 0
            # </editor-fold>

            result[self.tiling] = dict_tiles_counter_chunks

        save_json(result, self.counter_tiles_json)

    def heatmap(self):
        results = load_json(self.counter_tiles_json)

        for self.tiling in self.tiling_list:
            if self.tiling == '1x1': continue
            heatmap_tiling = self.get_tiles_folder / f'heatmap_tiling_{self.dataset_name}_{self.vid_proj}_{self.name}_{self.tiling}_fov{self.fov}.png'
            if heatmap_tiling.exists(): continue

            tiling_result = results[self.tiling]

            h, w = splitx(self.tiling)[::-1]
            grade = np.zeros((h*w,))

            for item in tiling_result: grade[int(item)] = tiling_result[item]
            grade = grade.reshape((h, w))

            fig, ax = plt.subplots()
            im = ax.imshow(grade, cmap='jet', )
            ax.set_title(f'Tiling {self.tiling}')
            fig.colorbar(im, ax=ax, label='chunk frequency')

            # fig.show()
            fig.savefig(f'{heatmap_tiling}')

    # def plot_test(self):
    #     # self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'|'chunks']: list[list[str]] | d
    #     self.results = load_json(self.get_tiles_json)
    #
    #     users_list = self.seen_tiles_metric[self.vid_proj][self.name]['1x1']['16'].keys()
    #
    #     for self.tiling in self.tiling_list:
    #         folder = self.seen_metrics_folder / f'1_{self.name}'
    #         folder.mkdir(parents=True, exist_ok=True)
    #
    #         for self.user in users_list:
    #             fig: plt.Figure
    #             fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
    #             ax = list(ax.flat)
    #             ax: list[plt.Axes]
    #
    #             for self.quality in self.quality_list:
    #                 result5 = defaultdict(list)    # By chunk
    #
    #                 for self.chunk in self.chunk_list:
    #                     seen_tiles_metric = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]
    #                     tiles_list = seen_tiles_metric['time'].keys()
    #
    #                     result5[f'n_tiles'].append(len(tiles_list))
    #                     for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
    #                         value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
    #                         percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
    #                         result5[f'{self.metric}_sum'].append(np.sum(value))         # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)
    #                         result5[f'{self.metric}_avg'].append(np.average(value))     # tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação dos tiles)
    #                         result5[f'{self.metric}_std'].append(np.std(value))
    #                         result5[f'{self.metric}_min'].append(percentile[0])
    #                         result5[f'{self.metric}_q1'].append(percentile[1])
    #                         result5[f'{self.metric}_median'].append(percentile[2])
    #                         result5[f'{self.metric}_q2'].append(percentile[3])
    #                         result5[f'{self.metric}_max'].append(percentile[4])
    #
    #                 ax[0].plot(result5['time_sum'], label=f'CRF{self.quality}')
    #                 ax[1].plot(result5['time_avg'], label=f'CRF{self.quality}')
    #                 ax[2].plot(result5['rate_sum'], label=f'CRF{self.quality}')
    #                 ax[3].plot(result5['PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[4].plot(result5['S-PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[5].plot(result5['WS-PSNR_avg'], label=f'CRF{self.quality}')
    #                 ax[6].plot(result5['n_tiles'], label=f'CRF{self.quality}')
    #
    #                 ax[0].set_title('time_sum')
    #                 ax[1].set_title('time_avg')
    #                 ax[2].set_title('rate_sum')
    #                 ax[3].set_title('PSNR_avg')
    #                 ax[4].set_title('S-PSNR_avg')
    #                 ax[5].set_title('WS-PSNR_avg')
    #                 ax[6].set_title('n_tiles')
    #
    #             for a in ax[:-1]:
    #                 a.legend(loc='upper right')
    #
    #             fig.suptitle(f'{self.video} {self.tiling} - user {self.user}')
    #             fig.suptitle(f'{self.video} {self.tiling} - user {self.user}')
    #             fig.tight_layout()
    #             # fig.show()
    #             img_name = folder / f'{self.tiling}_user{self.user}.png'
    #             fig.savefig(img_name)
    #             plt.close(fig)
    #


class UserProjectionMetricsProps(GetTilesProps, SegmentsQualityPaths):
    seen_tiles_metric: AutoDict
    _video: str
    _tiling: str

    time_data: dict
    rate_data: dict
    qlt_data: dict
    get_tiles_data: dict

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value

    @property
    def seen_metrics_folder(self) -> Path:
        folder = self.project_path / 'UserProjectionMetrics'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def seen_metrics_json(self) -> Path:
        return self.seen_metrics_folder / f'seen_metrics_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}.json'

    def get_get_tiles(self):
        try:
            tiles_list = self.get_tiles_data[self.vid_proj][self.name][self.tiling][self.user]['chunks'][self.chunk]
        except (KeyError, AttributeError):
            self.get_tiles_data = load_json(self.get_tiles_json, object_hook=dict)
            self.time_data = load_json(self.dectime_result_json, object_hook=dict)
            self.rate_data = load_json(self.bitrate_result_json, object_hook=dict)
            self.qlt_data = load_json(self.quality_result_json, object_hook=dict)
            tiles_list = self.get_tiles_data[self.vid_proj][self.name][self.tiling][self.user]['chunks'][self.chunk]
        return tiles_list


class UserProjectionMetrics(UserProjectionMetricsProps):
    def __init__(self, config):
        self.config = config
        self.print_resume()
        self.main()
        self.graphs1()
        self.graphs2()

    def main(self):
        for self.video in self.videos_list:
            if self.seen_metrics_json.exists(): continue

            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    for self.quality in self.quality_list:
                        for self.chunk in self.chunk_list:
                            self.worker()

            print(f'  Saving get tiles... ', end='')
            save_json(self.seen_tiles_metric, self.seen_metrics_json)
            print(f'  Finished.')

        print('')

    def worker(self):
        print(f'\r  Get Tiles - {self.vid_proj} {self.name} {self.tiling} - user{self.user} ... ', end='')

        for self.tile in self.get_get_tiles():
            dectime_val = self.time_data[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            bitrate_val = self.rate_data[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            quality_val = self.qlt_data[self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]

            try:
                metrics_result = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]
            except (NameError, AttributeError, KeyError):
                self.seen_tiles_metric = AutoDict()
                metrics_result = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]

            metrics_result['time'][self.tile] = float(np.average(dectime_val))
            metrics_result['rate'][self.tile] = float(bitrate_val)
            metrics_result['time_std'][self.tile] = float(np.std(dectime_val))
            metrics_result['PSNR'][self.tile] = quality_val['PSNR']
            metrics_result['WS-PSNR'][self.tile] = quality_val['WS-PSNR']
            metrics_result['S-PSNR'][self.tile] = quality_val['S-PSNR']

    def graphs1(self):
        # for each user plot quality in function of chunks
        def img_name():
            folder = self.seen_metrics_folder / f'1_{self.name}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.tiling}_user{self.user}.png'

        def loop_video_tiling_user():
            for self.video in self.videos_list:
                # for self.tiling in ['6x4']:
                for self.tiling in self.tiling_list:
                    for self.user in self.users_list:
                        yield 

        for _ in loop_video_tiling_user():
            if img_name().exists(): continue
            print(f'\r{img_name()}', end='')
            
            fig: plt.Figure
            ax: list[plt.Axes]
            fig, ax = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
            ax: plt.Axes
            ax = list(ax.flat)
            result_by_quality = AutoDict()  # By quality by chunk

            for self.quality in self.quality_list:
                for self.chunk in self.chunk_list:
                    # <editor-fold desc="get seen_tiles_metric">
                    try:
                        seen_tiles_metric = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]
                    except (KeyError, AttributeError):
                        self.seen_tiles_metric = load_json(self.seen_metrics_json)
                        seen_tiles_metric = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]
                    # </editor-fold>

                    tiles_list = seen_tiles_metric['time'].keys()
                    try:
                        result_by_quality[self.quality][f'n_tiles'].append(len(tiles_list))
                    except AttributeError:
                        result_by_quality[self.quality][f'n_tiles'] = [len(tiles_list)]

                    for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
                        tile_metric_value = [seen_tiles_metric[self.metric][tile] for tile in tiles_list]
                        percentile = list(np.percentile(tile_metric_value, [0, 25, 50, 75, 100]))
                        try:
                            result_by_quality[self.quality][f'{self.metric}_sum'].append(np.sum(tile_metric_value))         # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)
                        except AttributeError:
                            result_by_quality[self.quality] = defaultdict(list)
                            result_by_quality[self.quality][f'{self.metric}_sum'].append(np.sum(tile_metric_value))         # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)

                        result_by_quality[self.quality][f'{self.metric}_avg'].append(np.average(tile_metric_value))     # tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação dos tiles)
                        result_by_quality[self.quality][f'{self.metric}_std'].append(np.std(tile_metric_value))
                        result_by_quality[self.quality][f'{self.metric}_min'].append(percentile[0])
                        result_by_quality[self.quality][f'{self.metric}_q1'].append(percentile[1])
                        result_by_quality[self.quality][f'{self.metric}_median'].append(percentile[2])
                        result_by_quality[self.quality][f'{self.metric}_q2'].append(percentile[3])
                        result_by_quality[self.quality][f'{self.metric}_max'].append(percentile[4])

                ax[0].plot(result_by_quality[self.quality]['time_sum'], label=f'CRF{self.quality}')
                ax[1].plot(result_by_quality[self.quality]['time_avg'], label=f'CRF{self.quality}')
                ax[2].plot(result_by_quality[self.quality]['rate_sum'], label=f'CRF{self.quality}')
                ax[3].plot(result_by_quality[self.quality]['PSNR_avg'], label=f'CRF{self.quality}')
                ax[4].plot(result_by_quality[self.quality]['S-PSNR_avg'], label=f'CRF{self.quality}')
                ax[5].plot(result_by_quality[self.quality]['WS-PSNR_avg'], label=f'CRF{self.quality}')
                ax[6].plot(result_by_quality[self.quality]['n_tiles'], label=f'CRF{self.quality}')

            ax[0].set_title('Tempo de decodificação total')
            ax[1].set_title('Tempo médio de decodificação')
            ax[2].set_title('Taxa de bits total')
            ax[3].set_title(f'PSNR médio')
            ax[4].set_title('S-PSNR médio')
            ax[5].set_title('WS-PSNR médio')
            ax[6].set_title('Número de ladrilhos')
            for a in ax[:-2]: a.legend(loc='upper right')

            name = self.name.replace('_nas', '').replace('_', ' ').title()
            fig.suptitle(f'{name} {self.tiling} - user {self.user}')
            fig.tight_layout()
            fig.show()
            fig.savefig(img_name())
            plt.close(fig)

    def graphs2(self):
        def img_name():
            folder = self.seen_metrics_folder / f'2_aggregate'
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f'{self.name}_{self.tiling}.png'

        def loop_video_tiling():
            for self.video in self.videos_list:
                self.seen_tiles_metric = load_json(self.seen_metrics_json)
                for self.tiling in self.tiling_list:
                    yield

        # Compara usuários
        for _ in loop_video_tiling():
            if img_name().exists(): continue
            print(img_name(), end='')

            fig: plt.Figure
            ax: list[plt.Axes]
            fig, ax = plt.subplots(2, 5, figsize=(15, 5), dpi=200)
            ax: plt.Axes
            ax = list(ax.flat)

            for self.quality in self.quality_list:
                result_lv2 = defaultdict(list)    # By chunk

                for self.user in self.users_list:
                    result_lv1 = defaultdict(list)    # By chunk

                    for self.chunk in self.chunk_list:
                        seen_tiles_data = self.seen_tiles_metric[self.vid_proj][self.name][self.tiling][self.quality][self.user][self.chunk]
                        tiles_list = seen_tiles_data['time'].keys()

                        result_lv1[f'n_tiles'].append(len(tiles_list))
                        for self.metric in ['time', 'rate', 'PSNR', 'WS-PSNR', 'S-PSNR']:
                            value = [seen_tiles_data[self.metric][tile] for tile in tiles_list]
                            percentile = list(np.percentile(value, [0, 25, 50, 75, 100]))
                            result_lv1[f'{self.metric}_sum'].append(np.sum(value))         # Tempo total de um chunk (sem decodificação paralela) (soma os tempos de decodificação dos tiles)
                            result_lv1[f'{self.metric}_avg'].append(np.average(value))     # tempo médio de um chunk (com decodificação paralela) (média dos tempos de decodificação dos tiles)
                            result_lv1[f'{self.metric}_std'].append(np.std(value))
                            result_lv1[f'{self.metric}_min'].append(percentile[0])
                            result_lv1[f'{self.metric}_q1'].append(percentile[1])
                            result_lv1[f'{self.metric}_median'].append(percentile[2])
                            result_lv1[f'{self.metric}_q2'].append(percentile[3])
                            result_lv1[f'{self.metric}_max'].append(percentile[4])

                    # each metrics represent the metrics by complete reprodution of the one vídeo with one tiling in one quality for one user
                    result_lv2[f'time_total'].append(np.sum(result_lv1[f'time_sum']))         # tempo total sem decodificação paralela
                    result_lv2[f'time_avg_sum'].append(np.average(result_lv1[f'time_sum']))   # tempo médio sem decodificação paralela
                    result_lv2[f'time_total_avg'].append(np.sum(result_lv1[f'time_avg']))     # tempo total com decodificação paralela
                    result_lv2[f'time_avg_avg'].append(np.average(result_lv1[f'time_avg']))   # tempo total com decodificação paralela
                    result_lv2[f'rate_total'].append(np.sum(result_lv1[f'rate_sum']))         # taxa de bits sempre soma
                    result_lv2[f'psnr_avg'].append(np.average(result_lv1[f'PSNR_avg']))       # qualidade sempre é média
                    result_lv2[f'ws_psnr_avg'].append(np.average(result_lv1[f'WS-PSNR_avg']))
                    result_lv2[f's_psnr_avg'].append(np.average(result_lv1[f'S-PSNR_avg']))
                    result_lv2[f'n_tiles_avg'].append(np.average(result_lv1[f'n_tiles']))
                    result_lv2[f'n_tiles_total'].append(np.sum(result_lv1[f'n_tiles']))

                result4_df = pd.DataFrame(result_lv2)
                # result4_df = result4_df.sort_values(by=['rate_total'])
                x = list(range(len(result4_df['time_total'])))
                ax[0].bar(x, result4_df['time_total'], label=f'CRF{self.quality}')
                ax[1].bar(x, result4_df['time_avg_sum'], label=f'CRF{self.quality}')
                ax[2].bar(x, result4_df['time_total_avg'], label=f'CRF{self.quality}')
                ax[3].bar(x, result4_df['time_avg_avg'], label=f'CRF{self.quality}')
                ax[4].bar(x, result4_df['rate_total'], label=f'CRF{self.quality}')
                ax[5].bar(x, result4_df['psnr_avg'], label=f'CRF{self.quality}')
                ax[6].bar(x, result4_df['ws_psnr_avg'], label=f'CRF{self.quality}')
                ax[7].bar(x, result4_df['s_psnr_avg'], label=f'CRF{self.quality}')
                ax[8].bar(x, result4_df['n_tiles_avg'], label=f'CRF{self.quality}')
                ax[9].bar(x, result4_df['n_tiles_total'], label=f'CRF{self.quality}')

                ax[0].set_title('time_total')
                ax[1].set_title('time_avg_sum')
                ax[2].set_title('time_total_avg')
                ax[3].set_title('time_avg_avg')
                ax[4].set_title('rate_total')
                ax[5].set_title('psnr_avg')
                ax[6].set_title('ws_psnr_avg')
                ax[7].set_title('s_psnr_avg')
                ax[8].set_title('n_tiles_avg')
                ax[9].set_title('n_tiles_total')

            for a in ax[:-2]:
                a.legend(loc='upper right')

            fig.suptitle(f'{self.video} {self.tiling}')
            fig.tight_layout()
            # fig.show()
            fig.savefig(img_name)
            img_name = img_name().parent / f'{self.tiling}_{self.name}.png'
            fig.savefig(img_name)
            plt.close(fig)

            # result3[f'time_avg_total'].append(np.average(result4[f'time_total']))  # comparando entre usuários usamos o tempo médio
            # result3[f'time_avg_avg_sum'].append(np.sum(result4[f'time_avg_sum']))  # tempo médio sem paralelismo
            # result3[f'time_avg_avg'].append(np.average(result4[f'time_avg']))  # tempo total com decodificação paralela
            # result3[f'rate_total'].append(np.sum(result4[f'rate_sum']))  # taxa de bits sempre soma
            # result3[f'psnr_avg'].append(np.average(result4[f'PSNR_avg']))  # qualidade sempre é média
            # result3[f'ws_psnr_avg'].append(np.average(result4[f'WS-PSNR']))
            # result3[f's_psnr_avg'].append(np.average(result4[f'S-PSNR']))


class ViewportPSNRProps(GetTilesProps):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    _erp_list: dict[str, vp.ERP]
    _seen_tiles: dict
    _erp: vp.ERP
    dataset_data: dict
    erp_list: dict
    readers: dict
    seen_tiles: dict
    yaw_pitch_roll_frames: list
    video_frame_idx: int
    tile_h: float
    tile_w: float

    ## Lists #############################################
    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list

    ## Properties #############################################
    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def get_seen_tiles(self) -> dict[str, list]:
        """
        seen_tiles = self.seen_tiles[self.vid_proj][self.name][self.tiling][self.user]['chunks']

        :return:
        """
        while True:
            try:
                return self.seen_tiles[self.vid_proj][self.name][self.tiling][self.user]['chunks']
            except KeyError:
                del self._seen_tiles

    @property
    def seen_tiles(self):
        while True:
            try:
                return self._seen_tiles
            except AttributeError:
                self._seen_tiles = load_json(self.get_tiles_json)

    @property
    def n_frames(self):
        return int(self.duration) * int(self.fps)

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value

    @property
    def erp(self) -> vp.ERP:
        """
        self.erp_list[self.tiling]
        :return:
        """
        self._erp = self.erp_list[self.tiling]
        return self._erp

    @erp.setter
    def erp(self, value: vp.ERP):
        self._erp = value

    @property
    def erp_list(self) -> dict:
        """
        {tiling: vp.ERP(tiling, self.resolution, self.fov, vp_shape=np.array([90, 110]) * 6) for tiling in self.tiling_list}
        :return:
        """
        while True:
            try:
                return self._erp_list
            except AttributeError:
                print(f'Loading list of ERPs')
                self._erp_list = {tiling: vp.ERP(tiling, self.resolution, self.fov, vp_shape=np.array([90, 110]) * 6) for tiling in self.tiling_list}

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    ## Paths #############################################
    @property
    def viewport_psnr_path(self) -> Path:
        folder = self.project_path / 'ViewportPSNR'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def viewport_psnr_file(self) -> Path:
        folder = self.viewport_psnr_path / f'{self.vid_proj}_{self.name}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"user{self.user}_{self.tiling}.json"

    ## Methods #############################################
    def mount_frame(self, proj_frame, tiles_list, quality):
        self.quality = quality
        for self.tile in tiles_list:
            try:
                is_ok, tile_frame = self.readers[self.quality][self.tile].read()
            except (AttributeError, KeyError):
                self.readers = {self.quality: {self.tile: cv.VideoCapture(f'{self.segment_file}')}}
                is_ok, tile_frame = self.readers[self.quality][self.tile].read()

            m, n = idx2xy(idx=int(self.tile), shape=splitx(self.tiling)[::-1])
            tile_y, tile_x = self.erp.tile_res[-2] * n, self.erp.tile_res[-1] * m
            # tile_frame = cv.cvtColor(tile_frame, cv.COLOR_BGR2YUV)[:, :, 0]
            proj_frame[tile_y:tile_y + self.erp.tile_res[-2], tile_x:tile_x + self.erp.tile_res[-1], :] = tile_frame

    def output_exist(self, overwrite=False):
        if self.viewport_psnr_file.exists() and not overwrite:
            print(f'  The data file "{self.viewport_psnr_file}" exist.')
            return True
        return False

    @property
    def viewport_psnr_folder(self) -> Path:
        """
        Need None
        """
        folder = self.project_path / f'ViewportPSNR'
        folder.mkdir(parents=True, exist_ok=True)
        return folder


class ViewportPSNR(ViewportPSNRProps):
    seen_tiles_by_chunks: dict

    def __init__(self, config: str):
        self.config = Config(config)
        self.print_resume()
        self.main()

    def main(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    self.worker()
                    # self.make_video()

    @property
    def yaw_pitch_roll_frames(self):
        return self.dataset[self.name][self.user]

    def worker(self):
        if self.viewport_psnr_file.exists():
            print(f'The file {"/".join(self.viewport_psnr_file.parts[-2:])} exist. Skipping')
            return

        qlt_by_frame = AutoDict()
        proj_frame = np.zeros(tuple(self.erp.shape) + (3,), dtype='uint8')
        proj_frame_ref = np.zeros(tuple(self.erp.shape) + (3,), dtype='uint8')

        self.seen_tiles_by_chunks = self.seen_tiles[self.vid_proj][self.name][self.tiling][self.user]['chunks']

        for self.chunk in self.chunk_list:
            print(f'Processing {self.name}_{self.tiling}_user{self.user}_chunk{self.chunk}')

            seen_tiles = self.seen_tiles_by_chunks[self.chunk]
            seen_tiles = list(map(str, seen_tiles))
            proj_frame[:] = 0
            proj_frame_ref[:] = 0
            start = time.time()

            # Operations by frame
            for quality in self.quality_list:
                self.readers = {}
                for chunk_frame_idx in range(int(self.gop)):  # 30 frames per chunk
                    self.video_frame_idx = (int(self.chunk) - 1) * 30 + chunk_frame_idx
                    yaw_pitch_roll = self.yaw_pitch_roll_frames[self.video_frame_idx]

                    # <editor-fold desc=" Build 'proj_frame_ref' and 'proj_frame_ref' and get Viewport">
                    self.mount_frame(proj_frame_ref, seen_tiles, '0')
                    self.mount_frame(proj_frame, seen_tiles, quality)

                    viewport_frame_ref = self.erp.get_viewport(proj_frame_ref, yaw_pitch_roll)  # .astype('float64')
                    viewport_frame = self.erp.get_viewport(proj_frame, yaw_pitch_roll)  # .astype('float64')
                    # </editor-fold>

                    mse = np.average((viewport_frame_ref - viewport_frame) ** 2)

                    try:
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['mse'].append(mse)
                    except AttributeError:
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['mse'] = [mse]

                    print(f'\r    chunk{self.chunk}_crf{self.quality}_frame{chunk_frame_idx} - {time.time() - start: 0.3f} s', end='')
            print('')
        save_json(qlt_by_frame, self.viewport_psnr_file)

    def make_video(self):
        if self.tiling == '1x1': return
        # vheight, vwidth  = np.array([90, 110]) * 6
        width, height = 576, 288
        # yaw_pitch_roll_frames = self.dataset[self.name][self.user]

        def debug_img() -> Path:
            folder = self.viewport_psnr_path / f'{self.vid_proj}_{self.name}' / f"user{self.users_list[0]}_{self.tiling}"
            folder.mkdir(parents=True, exist_ok=True)
            return folder / f"frame_{self.video_frame_idx}.jpg"

        for self.chunk in self.chunk_list:
            print(f'Processing {self.name}_{self.tiling}_user{self.user}_chunk{self.chunk}')
            seen_tiles = list(map(str, self.seen_tiles_by_chunks[self.chunk]))
            proj_frame = np.zeros((2160, 4320, 3), dtype='uint8')
            self.readers = AutoDict()
            start = time.time()

            # Operations by frame
            for chunk_frame_idx in range(int(self.gop)):  # 30 frames per chunk
                self.video_frame_idx = (int(self.chunk) - 1) * 30 + chunk_frame_idx

                if debug_img().exists():
                    print(f'Debug Video exist. State=[{self.video}][{self.tiling}][user{self.user}]')
                    return

                yaw_pitch_roll = self.yaw_pitch_roll_frames[self.video_frame_idx]

                # Build projection frame and get viewport
                self.mount_frame(proj_frame, seen_tiles, '0')
                # proj_frame = cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)[:, :, 0]
                # Image.fromarray(proj_frame[..., ::-1]).show()
                # Image.fromarray(cv.cvtColor(proj_frame, cv.COLOR_BGR2RGB)).show()

                viewport_frame = self.erp.get_viewport(proj_frame, yaw_pitch_roll)  # .astype('float64')

                print(f'\r    chunk{self.chunk}_crf{self.quality}_frame{chunk_frame_idx} - {time.time() - start: 0.3f} s', end='')
                # vptiles = self.erp.get_vptiles(yaw_pitch_roll)

                # <editor-fold desc="Get and process frames">
                vp_frame_img = Image.fromarray(viewport_frame[..., ::-1])
                new_vwidth = int(np.round(height * vp_frame_img.width / vp_frame_img.height))
                vp_frame_img = vp_frame_img.resize((new_vwidth, height))

                proj_frame_img = Image.fromarray(proj_frame[..., ::-1])
                proj_frame_img = proj_frame_img.resize((width, height))
                # </editor-fold>

                cover_r = Image.new("RGB", (width, height), (255, 0, 0))
                cover_g = Image.new("RGB", (width, height), (0, 255, 0))
                cover_b = Image.new("RGB", (width, height), (0, 0, 255))
                cover_gray = Image.new("RGB", (width, height), (200, 200, 200))

                mask_all_tiles_borders = Image.fromarray(self.erp.draw_all_tiles_borders()).resize((width, height))
                mask_vp_tiles = Image.fromarray(self.erp.draw_vp_tiles(yaw_pitch_roll)).resize((width, height))
                mask_vp = Image.fromarray(self.erp.draw_vp_mask(yaw_pitch_roll, lum=200)).resize((width, height))
                mask_vp_borders = Image.fromarray(self.erp.draw_vp_borders(yaw_pitch_roll)).resize((width, height))

                frame_img = Image.composite(cover_r, proj_frame_img, mask=mask_all_tiles_borders)
                frame_img = Image.composite(cover_g, frame_img, mask=mask_vp_tiles)
                frame_img = Image.composite(cover_gray, frame_img, mask=mask_vp)
                frame_img = Image.composite(cover_b, frame_img, mask=mask_vp_borders)

                img_final = Image.new('RGB', (width + new_vwidth + 2, height), (0, 0, 0))
                img_final.paste(frame_img, (0, 0))
                img_final.paste(vp_frame_img, (width + 2, 0))
                # img_final.show()
                img_final.save(debug_img())

            print('')


class ViewportPSNRGraphs(ViewportPSNRProps):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    _quality: str
    dataset_data: dict
    dataset: dict
    erp_list: dict
    readers: AutoDict
    workfolder = None

    def __init__(self, config):
        self.config = config
        # self.workfolder = super().workfolder / 'viewport_videos'  # todo: fix it
        self.workfolder.mkdir(parents=True, exist_ok=True)

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    if self.output_exist(False): continue
                    # sse_frame = load_json(self.viewport_psnr_file)

                    for self.chunk in self.chunk_list:
                        for self.quality in self.quality_list:
                            for frame in range(int(self.fps)):  # 30 frames per chunk
                                self.worker()

    def worker(self, overwrite=False):
        pass

    @property
    def quality(self):
        return self._quality

    @quality.setter
    def quality(self, value):
        self._quality = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value
        self.yaw_pitch_roll_frames = self.dataset[self.name][self.user]

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self.get_tiles_data = load_json(self.get_tiles_json)
        self.erp_list = {tiling: vp.ERP(tiling, self.resolution, self.fov) for tiling in self.tiling_list}

    @property
    def tile(self):
        return self._tile

    @tile.setter
    def tile(self, value):
        self._tile = value

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value
        self.erp = self.erp_list[self.tiling]
        self.tile_h, self.tile_w = self.erp.tile_res[:2]

    def output_exist(self, overwrite=False):
        if self.viewport_psnr_file.exists() and not overwrite:
            print(f'  The data file "{self.viewport_psnr_file}" exist.')
            return True
        return False


# class CheckViewportPSNR(ViewportPSNR):
#
#     @property
#     def quality_list(self) -> list[str]:
#         quality_list: list = self.config['quality_list']
#         try:
#             quality_list.remove('0')
#         except ValueError:
#             pass
#         return quality_list
#
#     def loop(self):
#
#         self.workfolder.mkdir(parents=True, exist_ok=True)
#         self.sse_frame: dict = {}
#         self.frame: int = 0
#         self.log = []
#         debug1 = defaultdict(list)
#         debug2 = defaultdict(list)
#         # if self.output_exist(False): continue
#
#         for self.video in self.videos_list:
#             for self.tiling in self.tiling_list:
#                 for self.user in self.users_list:
#                     print(f'\r  Processing {self.vid_proj}_{self.name}_user{self.user}_{self.tiling}', end='')
#                     viewport_psnr_file = self.project_path / self.operation_folder / f'ViewportPSNR' / 'viewport_videos' / f'{self.vid_proj}_{self.name}' / f"user{self.user}_{self.tiling}.json"
#
#                     try:
#                         self.sse_frame = load_json(viewport_psnr_file)
#                     except FileNotFoundError:
#                         msg = f'FileNotFound: {self.viewport_psnr_file}'
#                         debug1['video'].append(self.video)
#                         debug1['tiling'].append(self.tiling)
#                         debug1['user'].append(self.user)
#                         debug1['msg'].append(msg)
#                         continue
#
#                     for self.quality in self.quality_list:
#                         psnr = self.sse_frame[self.vid_proj][self.name][self.tiling][self.user][self.quality]['psnr']
#                         n_frames = len(psnr)
#                         more_than_100 = [x for x in psnr if x > 100]
#
#                         if n_frames < (int(self.duration) * int(self.fps)):
#                             msg = f'Few frames {n_frames}.'
#                             debug2['video'].append(self.video)
#                             debug2['tiling'].append(self.tiling)
#                             debug2['user'].append(self.user)
#                             debug2['quality'].append(self.quality)
#                             debug2['error'].append('FrameError')
#                             debug2['msg'].append(msg)
#
#                         if len(more_than_100) > 0:
#                             msg = f'{len(more_than_100)} values above PSNR 100 - max={max(psnr)}'
#                             debug2['video'].append(self.video)
#                             debug2['tiling'].append(self.tiling)
#                             debug2['user'].append(self.user)
#                             debug2['quality'].append(self.quality)
#                             debug2['error'].append('ValueError')
#                             debug2['msg'].append(msg)
#
#         pd.DataFrame(debug1).to_csv("checkviewportpsnr1.csv", index=False)
#         pd.DataFrame(debug2).to_csv("checkviewportpsnr2.csv", index=False)
#
#         yield
#
#     def worker(self, **kwargs):
#         print(f'\rprocessing {self.vid_proj}_{self.name}_user{self.user}', end='')


class TestDataset(ViewportPSNRProps):
    def __init__(self, config):
        self.config = config
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for turn in range(self.decoding_num):
                        for self.tile in self.tiling_list:
                            for self.chunk in self.chunk_list:
                                print(f'Decoding {self.segment_file=}. {turn = }', end='')
                                self.worker()

    def worker(self, overwrite=False):
        for self.video in self.videos_list:
            proj_h, proj_w, n_channels = self.video_shape
            users_data = self.dataset[self.name]

            erp = vp.ERP(self.tiling, self.resolution, '110x90')
            get_tiles_data = load_json(self.get_tiles_json, object_hook=dict)

            for self.tiling in self.tiling_list:
                if self.tiling == '1x1': continue  # Remover depois

                folder = self.get_tiles_folder / f'{self.vid_proj}_{self.name}_{self.tiling}'
                folder.mkdir(parents=True, exist_ok=True)

                M, N = splitx(self.tiling)
                tile_w, tile_h = int(proj_w / M), int(proj_h / N)

                for user in self.users_list:
                    quality_list = ['0'] + self.quality_list

                    for self.quality in quality_list:

                        output_video = folder / f"user{user}_CRF{self.quality}.mp4"
                        if not overwrite and output_video.exists():
                            print(f'The output video {output_video} exist. Skipping')
                            continue

                        video_writer = FFmpegWriter(output_video, inputdict={'-r': '30'}, outputdict={'-crf': '0', '-r': '30', '-pix_fmt': 'yuv420p'})

                        yaw_pitch_roll_frames = iter(users_data[user])

                        for self.chunk in self.chunk_list:
                            get_tiles_val: list[int] = get_tiles_data[self.vid_proj][self.name][self.tiling][user]['chunks'][0][self.chunk]  # Foi um erro colocar isso na forma de lista. Remover o [0] um dia
                            tiles_reader: dict[str, FFmpegReader] = {str(self.tile): FFmpegReader(f'{self.segment_file}').nextFrame() for self.tile in get_tiles_val}
                            img_proj = np.zeros((proj_h, proj_w, n_channels), dtype='uint8')

                            for _ in range(30):  # 30 frames per chunk
                                fig, ax = plt.subplots(1, 2, figsize=(6.5, 2), dpi=200)
                                start = time.time()

                                # Build projection frame
                                for self.tile in map(str, get_tiles_val):
                                    tile_m, tile_n = idx2xy(int(self.tile), (N, M))
                                    tile_x, tile_y = tile_m * tile_w, tile_n * tile_h

                                    tile_frame = next(tiles_reader[self.tile])

                                    img_proj[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w, :] = tile_frame

                                print(f'Time to mount 1 frame {time.time() - start: 0.3f} s')

                                yaw, pitch, roll = next(yaw_pitch_roll_frames)
                                center_point = np.array([np.deg2rad(yaw), np.deg2rad(pitch)])  # camera center point (valid range [0,1])
                                # fov_video = nfov.toNFOV(img_proj, center_point).astype('float64')

                                start = time.time()
                                fov_ref = erp.get_viewport(img_proj, center_point)  # .astype('float64')
                                print(f'time to extract vp =  {time.time() - start: 0.3f} s')

                                ax[0].imshow(erp.projection)
                                ax[0].axis('off')
                                ax[1].imshow(fov_ref)
                                ax[1].axis('off')
                                plt.tight_layout()
                                plt.show()
                                plt.close()
                                video_writer.writeFrame(fov_ref)

                            video_writer.close()
                        print('')
                        video_writer.close()


UserMetricsOptions = {'0': ProcessNasrabadi,  # 0
                      '1': TestDataset,  # 1
                      '2': GetTiles,  # 2
                      '3': UserProjectionMetrics,  # 3
                      '4': ViewportPSNR,  # 4
                      }
