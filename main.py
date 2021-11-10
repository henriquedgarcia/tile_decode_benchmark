#!/usr/bin/env python3
from assets.dectime import QualityAssessment, TileDecodeBenchmark, CheckTileDecodeBenchmark
import logging

logging.basicConfig(level=logging.WARNING)


def main():
    # Config
    config_list = []
    config_list += [f'config/config_nas_cmp.json']
    # config_list += [f'config/config_nas_erp.json']
    # config_list += [f'config/config_test.json']
    # config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']

    x = True
    decode_time = []
    siti = []
    check_files = [x]
    results = []
    quality = []

    for config in config_list:
        if decode_time:
            TileDecodeBenchmark(config, 'PREPARE', overwrite=False)
            TileDecodeBenchmark(config, 'COMPRESS', overwrite=False)
            TileDecodeBenchmark(config, 'SEGMENT', overwrite=False)
            TileDecodeBenchmark(config, 'DECODE', overwrite=False)
            TileDecodeBenchmark(config, 'COLLECT_RESULTS', overwrite=False)
        if siti:
            # Measure normal SITI
            TileDecodeBenchmark(config, 'SITI', overwrite=False,
                                animate_graph=False, save=True)
        if check_files:
            # CheckTileDecodeBenchmark(config, 'CHECK_ORIGINAL', clean=False,
            #                          check_gop=False)
            # CheckTileDecodeBenchmark(config, 'CHECK_LOSSLESS', clean=False,
            #                          check_gop=False)
            CheckTileDecodeBenchmark(config, 'CHECK_COMPRESS', only_error=True,
                                     check_log=True, check_video=True,
                                     check_gop=False, clean=False)
            # CheckTileDecodeBenchmark(config, 'CHECK_SEGMENT', clean=False,
            #                          check_gop=False)
            # CheckTileDecodeBenchmark(config, 'CLEAN',
            #                          table_of_check=(
            #                              'CHECK_COMPRESS-table-2021-09-30 15:54:29.901483.csv'),
            #                          clean_log=True, clean_video=True,
            #                          video_of_log=False)
            # CheckTileDecodeBenchmark(config, 'CHECK_DECODE')
            # CheckTileDecodeBenchmark(config, 'CHECK_RESULTS')
        if results:
            # operation = dt.TileDecodeBenchmark(config)
            # operation.run('RESULTS', overwrite=False)
            # dta.HistByPattern(config).run(False)
            # dta.HistByPatternByQuality(config).run(False)
            # dta.BarByPatternByQuality(config).run(False)
            # dta.HistByPatternFullFrame(config).run(False)
            # dta.BarByPatternFullFrame(config).run(False)
            pass
        if quality:
            QualityAssessment(config, 'QUALITY_ALL', overwrite=False)
            QualityAssessment(config, 'RESULTS', overwrite=False)


if __name__ == '__main__':
    main()
    print('## Finished. ##')

