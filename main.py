#!/usr/bin/env python3
from lib.dectime import TileDecodeBenchmark, CheckTileDecodeBenchmark
import logging

logging.basicConfig(level=logging.ERROR)


def main():
    config_list = []
    # config_list += [f'config/config_nas_cmp.json']
    config_list += [f'config/config_nas_erp.json']
    # config_list += [f'config/config_test.json']
    # config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']

    x = True
    decode_time = [0, 5]  # 1-pre, 2-com, 3-seg, 4-dec, 5-collect
    check_files = [x, 7]  # 1-ori, 2-loss, 3-comp, 4-seg, 5-clean, 6-dec, 7-res
    siti = [0, 2]
    make_graphs = [0, 6]
    quality = [0, 5]

    for config in config_list:
        if decode_time[0]:
            opt = {
                1: ('PREPARE', dict(overwrite=False)),
                2: ('COMPRESS', dict(overwrite=False)),
                3: ('SEGMENT', dict(overwrite=False)),
                4: ('DECODE', dict(overwrite=False)),
                5: ('COLLECT_RESULTS', dict(overwrite=False)),
            }
            role_id = decode_time[1]
            role = opt[role_id][0]
            kwargs = opt[role_id][1]
            TileDecodeBenchmark(config=config, role=role, **kwargs)

        if check_files[0]:
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
                7: ('CHECK_RESULTS', dict(only_error=True, clean=False)),
            }
            role_id = check_files[1]
            role = opt[role_id][0]
            kwargs = opt[role_id][1]
            CheckTileDecodeBenchmark(config=config, role=role, **kwargs)

        if siti[0]:
            # Measure normal SITI
            TileDecodeBenchmark(config, 'SITI', overwrite=False,
                                animate_graph=False, save=True)

        if make_graphs[0]:
            # operation = dt.TileDecodeBenchmark(config)
            # operation.run('RESULTS', overwrite=False)
            # dta.HistByPattern(config).run(False)
            # dta.HistByPatternByQuality(config).run(False)
            # dta.BarByPatternByQuality(config).run(False)
            # dta.HistByPatternFullFrame(config).run(False)
            # dta.BarByPatternFullFrame(config).run(False)
            pass
        # if quality[0]:
        #     QualityAssessment(config, 'QUALITY_ALL', overwrite=False)
        #     QualityAssessment(config, 'RESULTS', overwrite=False)


if __name__ == '__main__':
    main()
    print('## Finished. ##')
