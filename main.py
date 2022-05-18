#!/usr/bin/env python3.9
import argparse
from lib.dectime import (TileDecodeBenchmark, CheckTiles, MakeViewport,
                         QualityAssessment, GetTiles, DectimeGraphs)
import logging

logging.basicConfig(level=logging.WARNING)

config_list = []
# config_list += [f'config/config_nas_erp_cmp.json']
# config_list += [f'config/config_nas_cmp.json']
config_list += [f'config/config_nas_erp.json']
# config_list += [f'config/config_test.json']
# config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']


def main():
    # decode_time(5)  # 1-pre, 2-com, 3-seg, 4-dec, 5-collect
    # check_files(8)  # 1-ori, 2-loss, 3-comp, 4-seg, 5-clean, 6-dec, 7-res

    # bins=['fd', 'rice', 'sturges', 'sqrt', 'doane', 'scott', 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
    # dectime_graphs(0)
    get_tiles(1)
    # make_viewport(0)

    # siti(2)
    # quality(5)  # 1-all, 2-psnr, 3-wspsnr, 4-spsnr, 5-results
    # tiles_from_dataset(2)  # 1-prepare, 2-get_tile
    pass


def dectime_graphs(role_id):
    role_list = ['BY_PATTERN', 'BY_PATTERN_BY_QUALITY',
                 'BY_PATTERN_FULL_FRAME', 'BY_VIDEO_BY_PATTERN_BY_QUALITY']
    DectimeGraphs(config, role=role_list[role_id], bins=[30], overwrite=False, n_dist=6)


def make_viewport(role_id):
    role_list = {0: ('NAS_ERP', 'nasrabadi_28videos'),
                 1: ('USER_ANALYSIS', 'nasrabadi_28videos')}
    MakeViewport(config,
                 role=role_list[role_id][0], ds_name=role_list[role_id][1],
                 overwrite=False)


def decode_time(role_id):
    role_list = ['PREPARE', 'COMPRESS', 'SEGMENT', 'DECODE', 'COLLECT_RESULTS']
    'results\\nasrabadi_28videos_6q_8p_erp\\get_tiles\\get_tiles_nasrabadi_28videos_petite_anse_erp_nas_1x1.json'

    TileDecodeBenchmark(config, role_list[role_id], overwrite=False)


def check_files(role_id):
    opt = {
        0: ('CHECK_ORIGINAL', dict(only_error=True, check_video=True,
                                   check_log=True, clean=False)),
        1: ('CHECK_LOSSLESS', dict(only_error=True, check_video=True,
                                   deep_check=True, clean=False)),
        2: ('CHECK_COMPRESS', dict(only_error=True, check_log=True,
                                   check_video=True, check_gop=False,
                                   clean=False)),
        3: ('CHECK_SEGMENT', dict(only_error=True, check_video=True,
                                  deep_check=False, clean=False)),
        4: ('CLEAN', dict(only_error=True, clean_log=True,
                          clean_video=True, video_of_log=False,
                          table_of_check=('CHECK_COMPRESS-table-2021-09'
                                          '-30 15:54:29.901483.csv'),
                          )),
        5: ('CHECK_DECODE', dict(only_error=True, clean=False)),
        6: ('CHECK_RESULTS', dict(only_error=True)),
        7: ('CHECK_GET_TILES', dict(dataset_name='nasrabadi_28videos')),
    }
    CheckTiles(config, opt[role_id][0], **opt[role_id][1])


def dashing(role_ini, role_end=None):
    opt = {
        1: ('PREPARE', dict(overwrite=False)),
        2: ('COMPRESS', dict(overwrite=False)),
        3: ('DASH', dict(overwrite=False)),
        4: ('MEASURE_CHUNKS', dict(overwrite=False)),
    }
    start(opt, TileDecodeBenchmark, role_ini, role_end)


def siti(role_ini, role_end=None):
    opt = {
        # Measure normal SITI
        1: ('SITI', dict(overwrite=False, animate_graph=False,
                         save=True)),
    }
    start(opt, TileDecodeBenchmark, role_ini, role_end)


def quality(role_ini, role_end=None):
    opt = {
        1: ('ALL', dict(overwrite=False)),
        2: ('PSNR', dict(overwrite=False)),
        3: ('WSPSNR', dict(overwrite=False)),
        4: ('SPSNR', dict(overwrite=False)),
        5: ('RESULTS', dict(overwrite=False)),
    }
    start(opt, QualityAssessment, role_ini, role_end)


def get_tiles(role_id):
    """
    ['PREPARE', 'GET_TILES', 'JSON2PICKLE', 'REFACTOR']
    :param role_id:
    :return:
    """
    role_list = ['PREPARE', 'GET_TILES', 'USER_ANALYSIS']

    GetTiles(config=config, role=role_list[role_id],
             overwrite=False, dataset_name='nasrabadi_28videos')


def start(opt, cls, role_ini, role_end=None):
    role_end = role_end if role_end else role_ini
    for i in range(role_ini, role_end + 1):
        role = opt[role_ini][0]
        kwargs = opt[role_ini][1]
        cls(config=config, role=role, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dectime Testbed')
    parser.add_argument('-c', '--config', default=config_list, nargs='*',
                        help='The path to config file',
                        )
    parser.add_argument('-r', '--role', default=None, nargs=2,
                        help='ROLE opt',
                        )
    args = parser.parse_args()

    for config in args.config:
        main()

    print('## Finished. ##')
