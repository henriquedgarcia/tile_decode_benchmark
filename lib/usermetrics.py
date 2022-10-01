import time
from abc import ABC
from enum import Enum
from itertools import count
from logging import warning
from pathlib import Path
from typing import Iterable, Union
from collections import defaultdict, Counter

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skvideo.io import FFmpegWriter, FFmpegReader

import lib.viewport as vp
from .assets2 import Base
from .qualityassessment import QualityAssessmentPaths
from .tiledecodebenchmark import TileDecodeBenchmarkPaths
from .util2 import load_json, save_json, splitx, AutoDict, cart2hcs, lin_interpol, idx2xy

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
    _workfolder: Path = None
    _csv_dataset_file: Path
    video_name: str
    user_id: str
    head_movement: pd.DataFrame

    @property
    def get_tiles_path(self) -> Path:
        folder = self.project_path / self.get_tiles_folder
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @property
    def workfolder(self) -> Path:
        """
        Need None
        :return:
        """
        if self._workfolder is None:
            folder = self.get_tiles_path / f'{self.__class__.__name__}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        else:
            return self._workfolder

    @workfolder.setter
    def workfolder(self, value):
        self._workfolder = value

    @property
    def dataset_folder(self) -> Path:
        return Path('datasets') / self.config["dataset_name"]

    @property
    def dataset_json(self) -> Path:
        return self.dataset_folder / f'{self.config["dataset_name"]}.json'

    @property
    def seen_metrics_json(self) -> Path:
        folder = self.project_path / 'seen_metrics'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f'seen_metrics_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}.json'

    @property
    def seen_tiles_json(self) -> Path:
        path = self.get_tiles_path / f'get_tiles_{self.config["dataset_name"]}_{self.vid_proj}_{self.name}_fov{self.fov}.json'
        return path

    @property
    def dataset_name(self):
        return self.config['dataset_name']

    @property
    def csv_dataset_file(self) -> Path:
        return self._csv_dataset_file

    @csv_dataset_file.setter
    def csv_dataset_file(self, value):
        self._csv_dataset_file = value
        user_nas_id, video_nas_id = self.csv_dataset_file.stem.split('_')
        self.video_name = self.video_id_map[video_nas_id]
        self.user_id = self.user_map[user_nas_id]

        names = ['timestamp', 'Qx', 'Qy', 'Qz', 'Qw', 'Vx', 'Vy', 'Vz']
        self.head_movement = pd.read_csv(self.csv_dataset_file, names=names)

    @property
    def viewport_psnr_file(self) -> Path:
        folder = self.workfolder / f'{self.vid_proj}_{self.name}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"user{self.user}_{self.tiling}.json"


class ProcessNasrabadi(GetTilesPath):
    csv_dataset_file: Path
    user: int
    user_map = {}
    dataset_final = AutoDict()
    _csv_dataset_file: Path
    previous_line: tuple
    frame_counter: int

    def loop(self):
        print(f'Processing dataset {self.dataset_folder}.')

        if self.output_exist(True): return

        self.video_id_map = load_json(f'{self.dataset_json.parent}/videos_map.json')
        self.user_map = load_json(f'{self.dataset_json.parent}/usermap.json')

        for self.csv_dataset_file in self.dataset_json.parent.glob('*/*.csv'):
            self.frame_counter = 0
            yield

        print(f'\nFinish. Saving as {self.dataset_json}.')
        save_json(self.dataset_final, self.dataset_json)

    def worker(self, overwrite=False):
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

    def output_exist(self, overwrite):
        if self.dataset_json.exists() and not overwrite:
            warning(f'The file {self.dataset_json} exist. Skipping.')
            return True
        return False

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
    dataset: dict
    _video: str
    user: int
    erp_list: dict
    results: AutoDict

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value

    @property
    def users_list(self) -> list[str]:
        return list(self.dataset[self.name].keys())

class GetTiles(GetTilesProps):
    def loop(self, overwrite=False):
        self.erp_list = {self.tiling: vp.ERP(self.tiling, '576x288', self.fov) for self.tiling in self.tiling_list}
        self.dataset = load_json(self.dataset_json)

        for self.video in self.videos_list:
            if self.seen_tiles_json.exists(): continue
            self.results = AutoDict()
            self.n_frames = int(self.duration) * int(self.gop)

            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    yield

    def worker(self):
        print(f'{self.name} - tiling {self.tiling} - User {self.user}')

        if self.tiling == '1x1':
            self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'] = [[0]] * self.n_frames
            self.results[self.vid_proj][self.name][self.tiling][self.user]['chunks'] = {i: [0] for i in range(1, int(self.duration) + 1)}
            return

        erp = self.erp_list[self.tiling]

        result_frames = []
        head_positions = []
        result_chunks = {}
        tiles_in_chunk = set()
        start = time.time()

        for frame, (yaw_pitch_roll) in enumerate(self.dataset[self.name][self.user]):
            erp.viewport.rotate(yaw_pitch_roll)
            vptiles: list[str] = erp.get_vptiles()
            tiles_in_chunk.update(vptiles)

            # vptiles by chunk
            next_is_new_chunk = (frame + 1) % 30 == 0
            if next_is_new_chunk:
                print(f'\r  user{self.user} - {frame:04d} - {time.time() - start:.3f}s - {str(tiles_in_chunk)[:25]: >25}', end='')
                chunk = frame//30 + 1  # start from 1 gpac defined
                result_chunks[f'{chunk}'] = list(map(str, tiles_in_chunk))
                tiles_in_chunk.clear()

            result_frames.append(vptiles)
            head_positions.append(yaw_pitch_roll)
        print('')

        self.saving(result_frames, head_positions, result_chunks)

    def saving(self, result_frames, head_positions, result_chunks):
        self.results[self.vid_proj][self.name][self.tiling][self.user]['frame'] = result_frames
        self.results[self.vid_proj][self.name][self.tiling][self.user]['head_positions'] = head_positions
        self.results[self.vid_proj][self.name][self.tiling][self.user]['chunks'] = result_chunks

        if self.tiling == self.tiling_list[-1] and self.user == self.users_list[-1]:
            print(f'Saving {self.seen_tiles_json}')
            save_json(self.results, self.seen_tiles_json)


class Heatmap(GetTilesPath):
    def loop(self):
        pass
    def worker(self):
        pass
    def user_analisys(self, overwrite=False):
        self.dataset = load_json(self.dataset_json)
        counter_tiles_json = self.get_tiles_path / f'counter_tiles_{self.dataset_name}.json'

        if counter_tiles_json.exists():
            result = load_json(counter_tiles_json)
        else:
            database = load_json(self.dataset_json, object_hook=dict)
            result = {}
            for self.tiling in self.tiling_list:
                # Collect tiling count
                tiles_counter = Counter()
                print(f'{self.tiling=}')
                nb_chunk = 0
                for self.video in self.videos_list:
                    users = database[self.name].keys()
                    get_tiles_json = self.get_tiles_path / f'get_tiles_{self.dataset_name}_{self.video}_{self.tiling}.json'
                    if not get_tiles_json.exists():
                        print(dict(tiles_counter))
                        break
                    print(f'  - {self.video=}')
                    get_tiles = load_json(get_tiles_json, object_hook=dict)
                    for user in users:
                        # hm = database[self.name][user]
                        chunks = get_tiles[self.vid_proj][self.tiling][user]['chunks'].keys()
                        for chunk in chunks:
                            seen_tiles_by_chunk = get_tiles[self.vid_proj][self.tiling][user]['chunks'][chunk]
                            tiles_counter = tiles_counter + Counter(seen_tiles_by_chunk)
                            nb_chunk += 1

                # normalize results
                dict_tiles_counter = dict(tiles_counter)
                column = []
                for tile_id in range(len(dict_tiles_counter)):
                    if not tile_id in dict_tiles_counter:
                        column.append(0.)
                    else:
                        column.append(dict_tiles_counter[tile_id] / nb_chunk)
                result[self.tiling] = column
                print(result)

            save_json(result, counter_tiles_json)

        # Create heatmap
        for self.tiling in self.tiling_list:
            tiling_result = result[self.tiling]
            shape = splitx(self.tiling)[::-1]
            grade = np.asarray(tiling_result).reshape(shape)
            fig, ax = plt.subplots()
            im = ax.imshow(grade, cmap='jet', )
            ax.set_title(f'Tiling {self.tiling}')
            fig.colorbar(im, ax=ax, label='chunk frequency')
            heatmap_tiling = self.get_tiles_path / f'heatmap_tiling_{self.dataset_name}_{self.tiling}.png'
            fig.savefig(f'{heatmap_tiling}')
            fig.show()


class TestViewport(GetTilesPath):
    _tiling: str
    _video: str
    _user: str
    get_tiles_type: str
    dataset_data: dict
    erp_list: dict

    def loop(self, get_tiles_type='chunk'):
        self.get_tiles_type = get_tiles_type
        self.workfolder = self.workfolder / f'video_teste_by_{get_tiles_type}'
        self.workfolder.mkdir(parents=True, exist_ok=True)

        self.dataset_data = load_json(self.dataset_json)
        self.erp_list = {tiling: vp.ERP(tiling, '576x288', self.fov) for tiling in self.tiling_list}

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                if self.tiling == '1x1': continue
                for self.user in self.users_list:
                    if self.output_exist(True): continue
                    yield

    def worker(self):
        print(f'{self.name} - tiling {self.tiling} - User {self.user}')
        writer = self.writer
        for frame_id, proj_frame, yaw_pitch_roll in zip(count(), self.get_reader, self.user_data):
            print(f'\rDrawing viewport. Frame {frame_id:04d}. ', end='')

            self.erp.viewport.rotate(yaw_pitch_roll)
            height, width = self.erp.shape

            frame_img = Image.fromarray(proj_frame).resize((width, height))

            # Draw all tiles border
            self.erp.clear_projection()
            self.erp.draw_all_tiles_borders(lum=255)
            cover = Image.new("RGB", (width, height), (255, 0, 0))
            frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(self.erp.projection))

            # Draw VP tiles by chunk
            iterator = (self.get_tiles_chunk[str(frame_id // 30 + 1)]
                        if self.get_tiles_type == 'chunk'
                        else self.get_tiles_frame[frame_id])
            self.erp.clear_projection()
            _ = [self.erp.draw_tile_border(idx=tile, lum=255) for tile in iterator]
            cover = Image.new("RGB", (width, height), (0, 255, 0))
            frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(self.erp.projection))

            # Draw viewport
            self.erp.clear_projection()
            self.erp.draw_vp(lum=200)
            cover = Image.new("RGB", (width, height), (200, 200, 200))
            frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(self.erp.projection))

            # Draw viewport borders
            self.erp.clear_projection()
            self.erp.draw_vp_borders(lum=255)
            cover = Image.new("RGB", (width, height), (0, 0, 255))
            frame_img = Image.composite(cover, frame_img, mask=Image.fromarray(self.erp.projection))

            print(f'The tiles {self.erp.get_vptiles()} have pixel in viewport.')
            vp_image = self.erp.get_viewport(np.asarray(proj_frame))

            vp_height, vp_width = self.erp.vp_shape
            new_vp_width = int(np.round(height * vp_width / vp_height))
            new_vp_height = height
            vp_image = Image.fromarray(vp_image).resize((new_vp_width, new_vp_height))

            new_im: Union[Iterable, Image.Image] = Image.new('RGB', (width + new_vp_width + 4, height), (255, 255, 255))
            new_im.paste(frame_img, (0, 0))
            new_im.paste(vp_image, ((width + 2), 0))

            # new_im.show()
            writer.writeFrame(np.asarray(new_im))
            # todo: remove this in the future
            # if frame_id >= 30: break

        print('')
        writer.close()

    def output_exist(self, overwrite=False):
        if self.output_video.exists() and not overwrite:
            print(f'  The data file "{self.output_video}" exist. Loading date.')
            return True
        return False

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self.get_tiles_data = load_json(self.seen_tiles_json)
        self.users_list = list(self.dataset_data[self.name].keys())

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value
        self.erp = self.erp_list[value]

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value
        self.get_tiles_frame = self.get_tiles_data[self.vid_proj][self.name][self.tiling][self.user]['frame']
        self.get_tiles_chunk = self.get_tiles_data[self.vid_proj][self.name][self.tiling][self.user]['chunks']
        self.user_data = self.dataset_data[self.name][self.user]

    @property
    def output_video(self):
        return self.workfolder / f'{self.name}_{self.user}_{self.tiling}.mp4'

    @property
    def writer(self):
        return FFmpegWriter(self.output_video,
                            inputdict={'-r': '30'},
                            outputdict={'-r': '30', '-pix_fmt': 'yuv420p'})

    @property
    def get_reader(self):
        basename = Path(f'{self.name}_{self.resolution}_{self.config["fps"]}_'
                        f'1x1_crf28')
        folder = self.project_path / self.compressed_folder / basename
        input_video = folder / f'tile0.mp4'
        return FFmpegReader(f'{input_video}').nextFrame()


class ViewportPSNRProps(GetTilesPath):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    quality: str
    dataset_data: dict
    dataset: dict
    erp_list: dict
    readers: AutoDict
    seen_tiles: dict
    yaw_pitch_roll_frames: list

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
        self.seen_tiles_by_chunks = self.seen_tiles[self.vid_proj][self.name][self.tiling][self.user]['chunks']

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self.n_frames = int(self.duration) * int(self.fps)
        self.seen_tiles = load_json(self.seen_tiles_json)
        # self.erp_list = {tiling: vp.ERP(tiling, self.resolution, self.fov) for tiling in self.tiling_list}
        self.erp_list = {tiling: vp.ERP(tiling, self.resolution, self.fov, vp_shape=np.array([90,110])*6) for tiling in self.tiling_list}

    @property
    def users_list(self):
        try:
            return list(self.dataset[self.name])
        except AttributeError:
            self.dataset = load_json(self.dataset_json)
            return list(self.dataset[self.name])

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

    @property
    def quality_list(self) -> list[str]:
        quality_list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list


class ViewportPSNR(ViewportPSNRProps):
    def loop(self):
        self.dataset = load_json(self.dataset_json)

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    yield

    def worker(self):
        qlt_by_frame = AutoDict()
        # proj_frame = np.zeros(self.video_shape[:2], dtype='uint8')
        # proj_frame_ref = np.zeros(self.video_shape[:2], dtype='uint8')
        proj_frame = np.zeros(tuple(self.erp.shape) + (3,), dtype='uint8')
        proj_frame_ref = np.zeros(tuple(self.erp.shape) + (3,), dtype='uint8')

        for self.chunk in self.chunk_list:
            print(f'Processing {self.name}_{self.tiling}_user{self.user}_chunk{self.chunk}')

            seen_tiles = self.seen_tiles_by_chunks[self.chunk]
            seen_tiles = list(map(str, seen_tiles))
            proj_frame[:] = 0
            proj_frame_ref[:] = 0
            start = time.time()

            # Operations by frame
            for quality in self.quality_list:
                self.readers = AutoDict()
                for chunk_frame_id in range(int(self.gop)):  # 30 frames per chunk
                    video_frame_id = (int(self.chunk)-1)*30 + chunk_frame_id

                    # Build reference frame and get vp
                    yaw_pitch_roll = self.yaw_pitch_roll_frames[video_frame_id]
                    self.erp.viewport.rotate(yaw_pitch_roll)

                    self.mount_frame(proj_frame_ref, seen_tiles, '0')
                    viewport_frame_ref = self.erp.get_viewport(proj_frame_ref)  # .astype('float64')

                    self.mount_frame(proj_frame, seen_tiles, quality)
                    viewport_frame = self.erp.get_viewport(proj_frame)  # .astype('float64')

                    mse = np.average((viewport_frame_ref - viewport_frame) ** 2)
                    psnr = 10 * np.log10((255. ** 2 / mse))

                    try:
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['psnr'].append(psnr)
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['mse'].append(mse)
                    except AttributeError:
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['psnr'] = [psnr]
                        qlt_by_frame[self.vid_proj][self.name][self.tiling][self.user][quality]['mse'] = [mse]

                    print(f'\r    chunk{self.chunk}_crf{self.quality}_frame{chunk_frame_id} - {time.time()-start: 0.3f} s', end='')

            print('')

        save_json(qlt_by_frame, self.viewport_psnr_file)
        # video_writer.close()

    def mount_frame(self, proj_frame, tiles_list, quality):
        self.quality = quality
        for self.tile in tiles_list:
            try:
                tile_frame = self.readers[self.quality][self.tile].read()[1]
            except AttributeError:
                try:
                    # print(f'    Loading user{self.user}_{self.segment_file.parents[0].name}/{self.segment_file.name} - {self.segment_file.stat().st_size: ,} bytes')
                    self.readers[self.quality][self.tile] = cv.VideoCapture(f'{self.segment_file}')
                    tile_frame = self.readers[self.quality][self.tile].read()[1]
                except FileNotFoundError:
                    print(f'    WARNING: The segment {self.segment_file} not found. Skipping')
                    continue
            tile_y, tile_x = self.erp.tiles_position[(int(self.tile))]
            # tile_frame = cv.cvtColor(tile_frame, cv.COLOR_BGR2YUV)[:, :, 0]
            # proj_frame[tile_y:tile_y + self.tile_h, tile_x:tile_x + self.tile_w] = tile_frame
            proj_frame[tile_y:tile_y + self.tile_h, tile_x:tile_x + self.tile_w,:] = tile_frame

    def output_exist(self, overwrite=False):
        if self.viewport_psnr_file.exists() and not overwrite:
            print(f'  The data file "{self.viewport_psnr_file}" exist.')
            return True
        return False


class ViewportPSNRGraphs(GetTilesPath):
    _tiling: str
    _video: str
    _tile: str
    _user: str
    _quality: str
    dataset_data: dict
    dataset: dict
    erp_list: dict
    readers: AutoDict

    def loop(self):
        self.workfolder = self.workfolder / 'viewport_videos'
        self.workfolder.mkdir(parents=True, exist_ok=True)
        self.dataset = load_json(self.dataset_json)

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    if self.output_exist(False): continue
                    sse_frame = load_json(self.viewport_psnr_file)

                    for self.chunk in self.chunk_list:
                        for self.quality in self.quality_list:
                            for frame in range(int(self.fps)):  # 30 frames per chunk
                                yield

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
        self.get_tiles_data = load_json(self.seen_tiles_json)
        self.users_list = list(self.dataset[self.name])
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


class CheckViewportPSNR(ViewportPSNR):
    @property
    def quality_list(self) -> list[str]:
        quality_list:list = self.config['quality_list']
        try:
            quality_list.remove('0')
        except ValueError:
            pass
        return quality_list

    def loop(self):
        from itertools import dropwhile
        self.workfolder.mkdir(parents=True, exist_ok=True)
        self.sse_frame: dict = {}
        self.frame: int = 0
        self.log = []
        debug1 = defaultdict(list)
        debug2 = defaultdict(list)
        # if self.output_exist(False): continue

        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.user in self.users_list:
                    print(f'\r  Processing {self.vid_proj}_{self.name}_user{self.user}_{self.tiling}', end='')
                    viewport_psnr_file = self.project_path / self.get_tiles_folder / f'ViewportPSNR' / 'viewport_videos' / f'{self.vid_proj}_{self.name}' / f"user{self.user}_{self.tiling}.json"

                    try:
                        self.sse_frame = load_json(viewport_psnr_file)
                    except FileNotFoundError:
                        msg = f'FileNotFound: {self.viewport_psnr_file}'
                        debug1['video'].append(self.video)
                        debug1['tiling'].append(self.tiling)
                        debug1['user'].append(self.user)
                        debug1['msg'].append(msg)
                        continue

                    for self.quality in self.quality_list:
                        psnr = self.sse_frame[self.vid_proj][self.name][self.tiling][self.user][self.quality]['psnr']
                        n_frames = len(psnr)
                        more_than_100 = [x for x in psnr if x > 100]

                        if n_frames < (int(self.duration) * int(self.fps)):
                            msg = f'Few frames {n_frames}.'
                            debug2['video'].append(self.video)
                            debug2['tiling'].append(self.tiling)
                            debug2['user'].append(self.user)
                            debug2['quality'].append(self.quality)
                            debug2['error'].append('FrameError')
                            debug2['msg'].append(msg)

                        if len(more_than_100) > 0:
                            msg = f'{len(more_than_100)} values above PSNR 100 - max={max(psnr)}'
                            debug2['video'].append(self.video)
                            debug2['tiling'].append(self.tiling)
                            debug2['user'].append(self.user)
                            debug2['quality'].append(self.quality)
                            debug2['error'].append('ValueError')
                            debug2['msg'].append(msg)

        pd.DataFrame(debug1).to_csv("checkviewportpsnr1.csv", index=False)
        pd.DataFrame(debug2).to_csv("checkviewportpsnr2.csv", index=False)

        yield

    def worker(self, **kwargs):
        print(f'\rprocessing {self.vid_proj}_{self.name}_user{self.user}', end='')


class ViewportMetrics(GetTilesPath, QualityAssessmentPaths):
    seen_tiles_data: AutoDict
    _video: str = None
    users: list = None
    user: int = None

    def output_exist(self):
        if self.seen_metrics_json.exists() and not self.overwrite:
            print(f'  The data file "{self.seen_tiles_json}" exist. Loading date.')
            return True
        return False

    @property
    def video(self):
        return self._video

    @video.setter
    def video(self, value):
        self._video = value
        self.time_data = load_json(self.dectime_result_json, object_hook=dict)
        self.rate_data = load_json(self.bitrate_result_json, object_hook=dict)
        self.qlt_data = load_json(self.quality_result_json, object_hook=dict)

    @property
    def tiling(self):
        return self._tiling

    @tiling.setter
    def tiling(self, value):
        self._tiling = value
        if self.video is None:
            warning('self.video is not assigned.')
        else:
            self.get_tiles_data = load_json(self.seen_tiles_json, object_hook=dict)
            self.users = list(self.get_tiles_data[self.vid_proj][self.name][self.tiling].keys())

    def loop(self):
        print('\n====== Get tiles data ======')

        self.seen_tiles_data = AutoDict()

        for self.video in self.videos_list:
            if self.output_exist(): return

            for self.tiling in self.tiling_list:
                print(f'\r  Get Tiles - {self.vid_proj}  {self.name} {self.tiling} - {len(self.users)} users ... ', end='')
                for self.user in self.users:
                    for self.chunk in self.chunk_list:
                        for self.quality in self.quality_list:
                            for self.tile in self.tile_list:
                                yield

    def worker(self):
        get_tiles = self.get_tiles_data[self.vid_proj][self.name][self.tiling][self.user]['chunks'][self.chunk]

        if int(self.tile) in get_tiles:
            dectime_val = self.time_data[self.vid_proj][self.tiling][self.quality][self.tile][self.chunk]
            bitrate_val = self.rate_data[self.vid_proj][self.tiling][self.quality][self.tile][self.chunk]
            quality_val = self.qlt_data[self.vid_proj][self.tiling][self.quality][self.tile][self.chunk]

            seen_tiles_result = self.seen_tiles_data[f'{self.user}'][self.vid_proj][self.name][self.tiling][self.quality][self.tile][self.chunk]
            seen_tiles_result['time'] = float(np.average(dectime_val['times']))
            seen_tiles_result['rate'] = float(bitrate_val['rate'])
            seen_tiles_result['time_std'] = float(np.std(dectime_val['times']))

            for metric in ['PSNR', 'WS-PSNR', 'S-PSNR']:
                value = quality_val[metric]
                if value == float('inf'):
                    value = 1000
                seen_tiles_result[metric] = value
                print('OK')

        print(f'  Saving get tiles... ', end='')
        # todo: criar arquivo para esse módulo
        # save_json(self.seen_tiles_data, self.seen_tiles_json)
        print(f'  Finished.')


class TestDataset(GetTilesPath):
    def loop(self):
        for self.video in self.videos_list:
            for self.tiling in self.tiling_list:
                for self.quality in self.quality_list:
                    for turn in range(self.decoding_num):
                        for self.tile in self.tiling_list:
                            for self.chunk in self.chunk_list:
                                print(f'Decoding {self.segment_file=}. {turn = }', end='')
                                yield

    def worker(self, overwrite=False):
        database = load_json(self.dataset_json)

        for self.video in self.videos_list:
            proj_h, proj_w, n_channels = self.video_shape
            users_data = database[self.name]

            erp = vp.ERP(self.tiling, self.resolution, '110x90')
            get_tiles_data = load_json(self.seen_tiles_json, object_hook=dict)

            for self.tiling in self.tiling_list:
                if self.tiling == '1x1': continue  # Remover depois

                folder = self.get_tiles_path / f'{self.vid_proj}_{self.name}_{self.tiling}'
                folder.mkdir(parents=True, exist_ok=True)

                users_list = get_tiles_data[self.vid_proj][self.name][self.tiling].keys()

                M, N = splitx(self.tiling)
                tile_w, tile_h = int(proj_w / M), int(proj_h / N)

                for user in users_list:
                    quality_list = ['0'] + self.quality_list

                    for self.quality in quality_list:

                        output_video = folder / f"user{user}_CRF{self.quality}.mp4"
                        if not overwrite and output_video.exists():
                            print(f'The output video {output_video} exist. Skipping')
                            continue

                        video_writer = FFmpegWriter(output_video, inputdict={'-r': '30'}, outputdict={'-crf': '0', '-r': '30', '-pix_fmt': 'yuv420p'})

                        yaw_pitch_roll_frames = iter(users_data[user])

                        for self.chunk in self.chunk_list:
                            get_tiles_val: list[int] = get_tiles_data[self.vid_proj][self.name][self.tiling][user]['chunks'][0][
                                self.chunk]  # Foi um erro colocar isso na forma de lista. Remover o [0] um dia

                            tiles_reader: dict[str, FFmpegReader] = {str(self.tile): FFmpegReader(f'{self.segment_file}').nextFrame() for self.tile in
                                                                     get_tiles_val}
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


class UserMetricsOptions(Enum):
    PROCESS_NASRABADI = 0
    TEST_DATASET = 1
    GET_TILES = 2
    TEST_VIEWPORT_VIDEO = 3
    PSNR_FROM_VIEWPORT = 4
    CHECK_VIEWPORT_PSNR = 5
    USER_ANALISYS = 6

    def __repr__(self):
        return str({self.value: self.name})



class UserMetrics(Base):
    operations = {'PROCESS_NASRABADI': ProcessNasrabadi,
                  'TEST_DATASET': TestDataset,
                  'GET_TILES': GetTiles,
                  'TEST_VIEWPORT_VIDEO': TestViewport,
                  'PSNR_FROM_VIEWPORT': ViewportPSNR,
                  'CHECK_VIEWPORT_PSNR': CheckViewportPSNR,
                  'USER_ANALISYS': TestDataset,
                  }
