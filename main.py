from dectime.dectime import Dectime, Role


def main():
    config = f'config/config_user_dectime_28videos_nas.json'

    tile_dectime = Dectime(config)

    tile_dectime.run(Role.PREPARE)
    tile_dectime.run(Role.COMPRESS)
    tile_dectime.run(Role.SEGMENT)
    # tile_dectime.run(Role.DECODE)
    # tile_dectime.run(Role.RESULTS)


if __name__ == '__main__':
    main()
