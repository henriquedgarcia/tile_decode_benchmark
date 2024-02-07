#!/usr/bin/env python3.9
import argparse
from typing import Type
import lib

# config = f'config/config_nas_erp.json'
# config = f'config/config_nas_cmp.json'

# config = f'config/config_nas_erp_cmp_qp.json'
config = f'config/config_nas_erp_cmp.json'

worker_list: dict[str, dict[str, Type]] = {
    '0': lib.TileDecodeBenchmarkOptions,
    # '1': ('CheckTiles', 'CheckTilesOptions'),
    '2': lib.DectimeGraphsOptions,
    '3': lib.QualityAssessmentOptions,
    # '4': ('MakeViewport', 'QualityAssessment'),
    # '5': ('Dashing', 'QualityAssessment'),
    # '6': ('QualityAssessment', 'QualityAssessment'),
    # '7': ('Siti', 'QualityAssessment'),
    '8': lib.UserMetricsOptions,
}


def make_help_txt():
    help_txt = ['Dectime Testbed.']
    help_txt += [f'| WORKER_ID: {{ROLE_ID: "Worker Name"}}, ...']
    for idx, worker in worker_list.items():
        line = f'| {repr(idx):^9}: '
        for idx2, opt in worker_list[idx].items():
            line += f'{{{idx2}: {opt.__name__}}},'
        help_txt += [line[:-1]]
    return '\n'.join(help_txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=make_help_txt())
    parser.add_argument('-c', default=None, metavar='CONFIG_FILE', help='The path to config file')
    parser.add_argument('-r', default=None, nargs=2, metavar=('WORKER_ID', 'ROLE_ID'), help=f'Two int separated by space.')
    args = parser.parse_args()

    worker_id, role_id = args.r
    config: str = args.c if args.c is not None else config

    worker_opt = worker_list[worker_id]
    role = worker_opt[role_id]
    role(config)

    print(f'\n The end of {role} ======')
