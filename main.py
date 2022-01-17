#!/usr/bin/env python3.9
from lib.dectime import (TileDecodeBenchmark, CheckTiles,
                         QualityAssessment, GetTiles)
import logging

logging.basicConfig(level=logging.WARNING)

config_list = []
# config_list += [f'config/config_nas_cmp.json']
# config_list += [f'config/config_nas_erp.json']
config_list += [f'config/config_test.json']
# config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']
x = True


def main():
    decode_time(0, 5)  # 1-pre, 2-com, 3-seg, 4-dec, 5-collect
    check_files(0, 7)  # 1-ori, 2-loss, 3-comp, 4-seg, 5-clean, 6-dec, 7-res
    siti(0, 2)
    make_graphs(0, 6)
    quality(0, 5)  # 1-all, 2-psnr, 3-wspsnr, 4-spsnr, 5-results
    tiles_from_dataset(x, 1)  # 1-prepare, 2-get_tile

def decode_time(run, role_id, role_end=None):
    opt = {
        1: ('PREPARE', dict(overwrite=False)),
        2: ('COMPRESS', dict(overwrite=False)),
        3: ('SEGMENT', dict(overwrite=False)),
        4: ('DECODE', dict(overwrite=False)),
        5: ('COLLECT_RESULTS', dict(overwrite=False)),
    }
    start(run, opt, TileDecodeBenchmark, role_id, role_end)

def check_files(run, role_id, role_end=None):
    opt = {
        1: ('CHECK_ORIGINAL', dict(only_error=True, check_video=True,
                                   check_log=True, clean=False)),
        2: ('CHECK_LOSSLESS', dict(only_error=True, check_video=True,
                                   deep_check=True, clean=False)),
        3: ('CHECK_COMPRESS', dict(only_error=True, check_log=True,
                                   check_video=True, check_gop=False,
                                   clean=False)),
        4: ('CHECK_SEGMENT', dict(only_error=True, check_video=True,
                                  deep_check=False, clean=False)),
        5: ('CLEAN', dict(only_error=True, clean_log=True,
                          clean_video=True, video_of_log=False,
                          table_of_check=('CHECK_COMPRESS-table-2021-09'
                                          '-30 15:54:29.901483.csv'),
                          )),
        6: ('CHECK_DECODE', dict(only_error=True, clean=False)),
        7: ('CHECK_RESULTS', dict(only_error=True)),
    }
    start(run, opt, CheckTiles, role_id, role_end)

def siti(run, role_id, role_end=None):
    opt = {
        # Measure normal SITI
        1: ('SITI', dict(overwrite=False, animate_graph=False,
                         save=True)),
    }
    start(run, opt, TileDecodeBenchmark, role_id, role_end)

def quality(run, role_id, role_end=None):
    opt = {
        1: ('ALL', dict(overwrite=False)),
        2: ('PSNR', dict(overwrite=False)),
        3: ('WSPSNR', dict(overwrite=False)),
        4: ('SPSNR', dict(overwrite=False)),
        5: ('RESULTS', dict(overwrite=False)),
    }
    start(run, opt, QualityAssessment, role_id, role_end)

def tiles_from_dataset(run, role_id, role_end=None):
    opt = {
        # Measure normal SITI
        1: ('PREPARE', dict(overwrite=False,
                            database_name='nasrabadi_28videos')),
        2: ('GET_TILES', dict(overwrite=False,
                              database_name='nasrabadi_28videos',
                              )),
    }
    start(run, opt, GetTiles, role_id, role_end)

def make_graphs(run, role_id, role_end=None):
    opt = {
        1: ('SITI', dict(overwrite=False, animate_graph=False,
                         save=True)),
    }
    start(run, opt, TileDecodeBenchmark, role_id, role_end)

def start(run, opt, cls, role_id, role_end = None):
    if not run: return
    role_end = role_end if role_end else role_id
    for i in range(role_id, role_end + 1):
        role = opt[role_id][0]
        kwargs = opt[role_id][1]
        cls(config=config, role=role, **kwargs)

if __name__ == '__main__':
    import argparse

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
