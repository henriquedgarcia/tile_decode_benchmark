#!/usr/bin/env python3.9
import argparse
import logging

from lib.dectime import (TileDecodeBenchmark,DectimeGraphs)
                         # ,CheckTiles, QualityAssessment, MakeViewport, Dashing, QualityAssessment, Siti)

logging.basicConfig(level=logging.WARNING)

# config_list = [f'config/config_test.json']
# config_list = [f'config/config_ffmpeg_crf_12videos_60s.json']
# config_list = [f'config/config_nas_erp_cmp.json']
# config_list = [f'config/config_nas_cmp.json']
config = f'config/config_nas_erp.json'

worker_list = {
    0: ['TileDecodeBenchmark', {0: 'PREPARE', 1: 'COMPRESS', 2: 'SEGMENT', 3: 'DECODE', 4: 'COLLECT_RESULTS'}],
    1: ['CheckTiles', {0: 'CHECK_ORIGINAL', 1: 'CHECK_LOSSLESS', 2: 'CHECK_COMPRESS', 3: 'CHECK_SEGMENT', 4: 'CLEAN'}],
    2: ['DectimeGraphs', {0: 'BY_PATTERN', 1: 'BY_PATTERN_BY_QUALITY', 2: 'BY_VIDEO_BY_PATTERN_BY_QUALITY', 3: 'BY_PATTERN_FULL_FRAME', 4: 'BY_VIDEO_BY_PATTERN_BY_TILE_BY_CHUNK', 5: 'BY_VIDEO_BY_PATTERN_BY_TILE_BY_QUALITY_BY_CHUNK',}],
    3: ['QualityAssessment', {0: 'ALL', 1: 'PSNR', 2: 'WSPSNR', 3: 'SPSNR', 4: 'RESULTS'}],
    4: ['MakeViewport', {0: 'NAS_ERP', 1: 'USER_ANALYSIS'}],
    5: ['Dashing', {0: 'PREPARE', 1: 'COMPRESS', 2: 'DASH', 3: 'MEASURE_CHUNKS'}],
    6: ['QualityAssessment', {0: 'PREPARE', 1: 'GET_TILES', 2: 'USER_ANALYSIS'}],
    7: ['Siti', {0: 'SITI'}]
}

help_txt = 'Dectime Testbed.\n'
help_txt += f'\nWORKER_ID  {"Worker Name":19}   {{ROLE_ID: \'Role Name\', ...}}'
help_txt += '\n' + '-'*9 + '  ' + '-'*19 + '   ' + '-'*95
for key in worker_list:
    help_txt += f'\n{str(key):>9}: {worker_list[key][0]:19} - {str(worker_list[key][1]):19}'

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=help_txt)
parser.add_argument('-c', default=None, nargs=1,metavar='CONFIG_FILE',
                    help='The path to config file')
parser.add_argument('-r', default=None, nargs=2, metavar=('WORKER_ID','ROLE_ID'),
                    help=f'Two int separated by space.')
args = parser.parse_args()

if args.c is not None:
    config = args.c

worker_id, role_id = map(int, args.r)
worker = eval(worker_list[worker_id][0])
role = worker_list[worker_id][1][role_id]

worker(config=config, role=role)

print('## Finished. ##')
