import os
import json
import math
import subprocess
import sys
from abc import ABC, abstractmethod
from logging import info, debug, critical
from numbers import Real
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, NamedTuple, Tuple, Union

import numpy as np
from scipy import ndimage


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class AbstractConfig(ABC):
    _config_data: dict = {}
    
    @abstractmethod
    def __init__(self, config_file: Union[Path, str]): 
        self.load_config(config_file)

    def load_config(self, config_file: Union[Path, str]):
        debug(f'Loading {config_file}.')
        
        config_file = Path(config_file)
        assert config_file.exists(), (f'AbstractConfig.load_config: '
                                      f'{config_file} not exist.')
        content = config_file.read_text()
        self._config_data.update(json.loads(content))

        for key in self._config_data:
            debug(f'Setting {key}')
            value = self._config_data[key]
            setattr(self, key, value)


def run_command(command: str, log_to_save: Union[str, Path], mode: str = 'w'):
    """
    Run a shell command with subprocess module with realtime output.
    :param command: A command string to run.
    :param log_to_save: A path-like to save the process output.
    :param mode: The write mode: 'w' or 'a'.
    :return: stdout.
    """
    debug(command)

    with open(log_to_save, mode, encoding='utf-8') as f:
        f.write(f'{command}\n')
        subprocess.run(command, shell=True, stdout=f,
                       stderr=subprocess.STDOUT, encoding='utf-8')


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


def update_dictionary(value, dictionary: AutoDict, key1: Hashable = None,
                      key2: Hashable = None, key3: Hashable = None,
                      key4: Hashable = None, key5: Hashable = None):
    dict_ = dictionary
    if key1:
        if key2:
            dict_ = dict_[key1]
        else:
            dict_[key1] = value
    if key2:
        if key3:
            dict_ = dict_[key2]
        else:
            dict_[key2] = value
    if key3:
        if key4:
            dict_ = dict_[key3]
        else:
            dict_[key3] = value
    if key4:
        if key5:
            dict_ = dict_[key4]
        else:
            dict_[key4] = value
    if key5:
        dict_[key5] = value

    return dictionary


def dishevel_dictionary(dictionary: dict, key1: Hashable = None,
                        key2: Hashable = None, key3: Hashable = None,
                        key4: Hashable = None, key5: Hashable = None) -> Any:
    disheveled_dictionary = dictionary
    if key1: disheveled_dictionary = disheveled_dictionary[key1]
    if key2: disheveled_dictionary = disheveled_dictionary[key2]
    if key3: disheveled_dictionary = disheveled_dictionary[key3]
    if key4: disheveled_dictionary = disheveled_dictionary[key4]
    if key5: disheveled_dictionary = disheveled_dictionary[key5]
    return disheveled_dictionary


def make_menu(options_txt: list) -> (list, str):
    options = [str(o) for o in range(len(options_txt))]
    menu_lines = ['Options:']
    menu_lines.extend([f'{o} - {text}'
                       for o, text in zip(options, options_txt)])
    menu_lines.append(':')
    menu_txt = '\n'.join(menu_lines)
    return options, menu_txt


def menu(options_txt: list) -> int:
    options, menu_ = make_menu(options_txt)

    c = None
    while c not in options:
        c = input(menu_)

    return int(c)


def menu2(options_dict: Dict[int, Any]):
    options = []
    text = f'Options:\n'
    for idx in options_dict:
        text += f'{idx} - {options_dict[idx]}\n'
        options.append(idx)
    text += f': '

    c = None
    while c not in options:
        c = input(text)
    return c


def rem_file(file) -> None:
    if os.path.isfile(file):
        os.remove(file)

class StatsData(NamedTuple):
    average: float = None
    std: float = None
    correlation: float = None
    min: float = None
    quartile1: float = None
    median: float = None
    quartile3: float = None
    max: float = None


def calc_stats(data1: Iterable, data2: Iterable) \
        -> Tuple[StatsData, StatsData]:
    # Percentiles & Correlation
    per = [0, 25, 50, 75, 100]
    corr = np.corrcoef((data1, data2))[1][0]

    # Struct statistics results
    percentile_data1 = np.percentile(data1, per).T
    stats_data1 = StatsData(average=np.average(data1),
                            std=float(np.std(data1)),
                            correlation=corr,
                            min=percentile_data1[0],
                            quartile1=percentile_data1[1],
                            median=percentile_data1[2],
                            quartile3=percentile_data1[3],
                            max=percentile_data1[4],
                            )

    percentile_data2 = np.percentile(data2, per).T
    stats_data2 = StatsData(average=np.average(data2),
                            std=float(np.std(data2)),
                            correlation=corr,
                            min=percentile_data2[0],
                            quartile1=percentile_data2[1],
                            median=percentile_data2[2],
                            quartile3=percentile_data2[3],
                            max=percentile_data2[4],
                            )

    return stats_data1, stats_data2


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    Usage: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx

    :param iterable: A iterable. ;-)
    :param n: The length of blocks.
    :param fillvalue: If "len(iterable)" is not multiple of "n", fill with this.
    :return: A tuple with "n" elements of "iterable".
    """
    from itertools import zip_longest
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def sobel(frame):
    """
    Apply 1st order 2D Sobel filter
    :param frame:
    :return:
    """
    sobx = ndimage.sobel(frame, axis=0)
    soby = ndimage.sobel(frame, axis=1)
    sob = np.hypot(sobx, soby)
    return sob


class CircularNumber(Real):
    _value: float
    ini_value: float
    end_value: float

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        v, t = self._check_cicle(v)
        self._value = v
        self.turn += t

    def __init__(self, v: Union[float, int], turn: int = 0, value_range=(0., 360.)):
        self.turn = turn
        self.ini_value = value_range[0]
        self.end_value = value_range[1]
        self.value = v

    def _check_cicle(self, v: Union[float]):
        turn = 0
        while True:
            if v >= self.end_value:
                v -= self.end_value
                turn += 1
            elif v < self.ini_value:
                v += self.end_value
                turn -= 1
            elif v < self.end_value or v >= self.ini_value:
                break
        return v, turn

    def __float__(self):
        return CircularNumber(float(self.value), turn=self.turn)

    def __trunc__(self) -> Any:
        return CircularNumber(int(self.value), turn=self.turn)

    def __floor__(self) -> Any:
        return CircularNumber(math.floor(self.value), turn=self.turn)

    def __ceil__(self) -> Any:
        return CircularNumber(math.ceil(self.value), turn=self.turn)

    def __round__(self, ndigits=0) -> Any:
        return CircularNumber(round(len(self), ndigits))

    def __len__(self):
        return self.turn * self.end_value + self.value

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CircularNumber):
            lt = len(self) < len(other)
        elif isinstance(other, (float, int)):
            lt = len(self) < other
        else:
            raise TypeError(f"unsupported operand type(s) for <: '{type(self)}' and '{type(other)}'")
        return lt

    def __le__(self, other: Any) -> bool:
        if isinstance(other, CircularNumber):
            lt = len(self) > len(other)
        elif isinstance(other, (float, int)):
            lt = len(self) > other
        else:
            raise TypeError(f"unsupported operand type(s) for <=: '{type(self)}' and '{type(other)}'")
        return lt

    def __eq__(self, other):
        if isinstance(other, CircularNumber):
            lt = len(self) == len(other)
        elif isinstance(other, (float, int)):
            lt = len(self) == other
        else:
            raise TypeError(f"unsupported operand type(s) for <=: '{type(self)}' and '{type(other)}'")
        return lt

    def __abs__(self):
        return len(self)

    def __floordiv__(self, other: Union[int, float]) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) // len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) // other)
        else:
            raise TypeError(f"unsupported operand type(s) for //: '{type(self)}' and '{type(other)}'")
        return cn

    def __rfloordiv__(self, other: Union[int, float]) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) // len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other // len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for //: '{type(other)}' and '{type(self)}'")
        return cn

    def __mod__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) % len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) % other)
        else:
            raise TypeError(f"unsupported operand type(s) for %: '{type(self)}' and '{type(other)}'")
        return cn

    def __rmod__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) % len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other % len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for %: '{type(other)}' and '{type(self)}'")
        return cn

    def __add__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) + len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")
        return cn

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) + len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(other)}' and '{type(self)}'")
        return cn

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) * len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")
        return cn

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) * len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: '{type(other)}' and '{type(self)}'")
        return cn

    if sys.version_info < (3, 0):
        def __div__(self, other: Any) -> Any:
            self.__floordiv__(other)

        def __rdiv__(self, other):
            self.__rfloordiv__(other)

    def __truediv__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) / len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) / other)
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(self)}' and '{type(other)}'")
        return cn

    def __rtruediv__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) / len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other / len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for /: '{type(other)}' and '{type(self)}'")
        return cn

    def __neg__(self) -> Any:
        return CircularNumber(-len(self))

    def __pos__(self) -> Any:
        return CircularNumber(len(self))

    def __pow__(self, exponent: Any) -> Any:
        return CircularNumber(len(self) ** exponent)

    def __rpow__(self, base: Any) -> Any:
        return CircularNumber(base ** len(self))

    def __hash__(self) -> int:
        return hash(self)

    def __repr__(self):
        return f'{str(self.value)} = {self.turn}x{self.end_value} = {len(self)}'
