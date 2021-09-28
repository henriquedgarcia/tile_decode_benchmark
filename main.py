#!/usr/bin/env python3
from assets.dectime import QualityAssessment, TileDecodeBenchmark, CheckTileDecodeBenchmark
import logging

logging.basicConfig(level=logging.WARNING)


def main():
    # Config
    config_list = []
    # config_list += [f'config/config_nas_cmp.json']
    # config_list += [f'config/config_nas_erp.json']
    config_list += [f'config/config_test.json']
    # config_list += [f'config/config_ffmpeg_crf_12videos_60s.json']

    decode_time = False
    siti = False
    check_files = True
    results = False
    quality = False

    for config in config_list:
        if decode_time:
            TileDecodeBenchmark(config, 'PREPARE', overwrite=False)
            TileDecodeBenchmark(config, 'COMPRESS', overwrite=False)
            TileDecodeBenchmark(config, 'SEGMENT', overwrite=False)
            TileDecodeBenchmark(config, 'DECODE', overwrite=False)
            TileDecodeBenchmark(config, 'COLLECT_RESULTS', overwrite=False)
        if siti:
            # Measure normal SITI
            TileDecodeBenchmark(config, 'SITI', overwrite=False, animate_graph=False, save=True)
        if check_files:
            CheckTileDecodeBenchmark(config, 'CHECK_ORIGINAL', clean=False, check_gop=False)
            CheckTileDecodeBenchmark(config, 'CHECK_LOSSLESS', clean=False, check_gop=False)
            CheckTileDecodeBenchmark(config, 'CHECK_COMPRESS', clean=False, check_gop=False)
            # CheckTileDecodeBenchmark(config, 'CHECK_SEGMENT', clean=False, check_gop=False)
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
    print('## Finish. ##')

