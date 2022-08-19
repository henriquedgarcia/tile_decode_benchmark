#!/usr/bin/env python3.9
import argparse
from enum import Enum
from typing import Union, Type, Iterable

from lib.tiledecodebenchmark import TileDecodeBenchmarkOptions, TileDecodeBenchmark
from lib.usermetrics import UserMetrics, UserMetricsOptions

config = f'config/config_nas_erp.json'

worker_list: dict[int, tuple[Type, Union[Type, Enum, Iterable]]] = {
    0: (TileDecodeBenchmark, TileDecodeBenchmarkOptions),
    8: (UserMetrics, UserMetricsOptions),
}


def make_help_txt():
    help_txt = 'Dectime Testbed.\n'
    help_txt += f'\nWORKER_ID  {"Worker Name":19}   {{ROLE_ID: \'Role Name\', ...}}'
    help_txt += '\n' + '-'*9 + '  ' + '-'*19 + '   ' + '-'*95
    for idx in worker_list:
        worker = worker_list[idx][0].__class__.__name__
        role = worker_list[idx][1]
        help_txt += f'\n{str(idx):9>}: {worker:19} - {str(list(role)):19}'
    return help_txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=make_help_txt())
    parser.add_argument('-c', default=None, nargs=1,metavar='CONFIG_FILE', help='The path to config file')
    parser.add_argument('-r', default=None, nargs=2, metavar=('WORKER_ID','ROLE_ID'),  help=f'Two int separated by space.')
    args = parser.parse_args()

    worker_id, role_id = map(int, args.r)
    config: str = args.c if args.c is not None else config

    worker_cls, role_enum = worker_list[worker_id]
    role = role_enum(role_id)

    worker_cls(conf=config, role=role)

    print('## Finished. ##')
