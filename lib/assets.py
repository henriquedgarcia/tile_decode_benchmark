import math
import sys
from numbers import Real
from typing import NamedTuple, Union, Any, Optional, Callable, Tuple
from enum import Enum


class PointBCS(NamedTuple):
    yaw: float
    pitch: float
    roll: float = 0.0


class PointHCS(NamedTuple):
    azimuth: float
    elevation: float


class Pixel(NamedTuple):
    x: float
    y: float
    value: Union[float, int] = 0


class Point:
    def __init__(self, x: Union[float, int], y: Union[float, int],
                 z: Union[float, int] = None):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        if self.z is None:
            return f'Point(x={self.x}, y={self.y})'
        else:
            return f'Point(x={self.x}, y={self.y}, z={self.z})'

    def __str__(self):
        if self.z is None:
            return f'({self.x}, {self.y})'
        else:
            return f'({self.x}, {self.y}, {self.z})'

    def __iter__(self):
        if self.z is None:
            return iter((self.x, self.y))
        else:
            return iter((self.x, self.y, self.z))

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class AutoDict(dict):
    def __missing__(self, key):
        self[key] = type(self)()
        return self[key]


class Role:
    def __init__(self, name: str, deep: int, init: Optional[Callable] = None,
                 operation: Optional[Callable] = None,
                 finish: Optional[Callable] = None):
        self.name = name.upper()
        self.deep = deep
        self.init = init if init is not None else self.stub
        self.operation = operation if callable(operation) else self.stub
        self.finish = finish if callable(finish) else self.stub

    def stub(self):
        ...


class Position(NamedTuple):
    x: float
    y: float
    z: float = None

    def __str__(self):
        string = f'({self.x}, {self.y}'
        if self.z is not None:
            string = string + f',{self.z}'
        string = string + ')'
        return string


class Resolution:
    W: int
    H: int
    shape: Tuple[int]

    def __init__(self, resolution: Union[str, tuple]):
        if isinstance(resolution, str):
            w, h = resolution.split('x')
            self.shape = h, w
        elif isinstance(resolution, tuple):
            self.shape = resolution

    @property
    def shape(self) -> tuple:
        return self.H, self.W

    @shape.setter
    def shape(self, shape: tuple):
        self.H = round(float(shape[0]))
        self.W = round(float(shape[1]))

    def __iter__(self):
        return iter((self.H, self.W))

    def __str__(self):
        return f'{self.W}x{self.H}'

    def __repr__(self):
        return f'{self.W}x{self.H}'

    def __truediv__(self, shape: tuple):
        if isinstance(shape, tuple) and len(shape) == 2:
            return Resolution((self.H / shape[0], self.W / shape[1]))


class Fov(Resolution):
    ...


class StatsData(NamedTuple):
    average: float = None
    std: float = None
    correlation: float = None
    min: float = None
    quartile1: float = None
    median: float = None
    quartile3: float = None
    max: float = None


class Dataframes(Enum):
    STATS_DATAFRAME = 'df_stats'
    FITTED_DATAFRAME = 'df_dist'
    PAPER_DATAFRAME = 'df_paper'
    DATA_DATAFRAME = 'df_data'


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

    def __init__(self, v: Union[float, int], turn: int = 0,
                 value_range=(0., 360.)):
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
            raise TypeError(f"unsupported operand type(s) for <: "
                            f"'{type(self)}' and '{type(other)}'")
        return lt

    def __le__(self, other: Any) -> bool:
        if isinstance(other, CircularNumber):
            lt = len(self) > len(other)
        elif isinstance(other, (float, int)):
            lt = len(self) > other
        else:
            raise TypeError(f"unsupported operand type(s) for <=: "
                            f"'{type(self)}' and '{type(other)}'")
        return lt

    def __eq__(self, other):
        if isinstance(other, CircularNumber):
            lt = len(self) == len(other)
        elif isinstance(other, (float, int)):
            lt = len(self) == other
        else:
            raise TypeError(f"unsupported operand type(s) for <=: "
                            f"'{type(self)}' and '{type(other)}'")
        return lt

    def __abs__(self):
        return len(self)

    def __floordiv__(self, other: Union[int, float]) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) // len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) // other)
        else:
            raise TypeError(f"unsupported operand type(s) for //: "
                            f"'{type(self)}' and '{type(other)}'")
        return cn

    def __rfloordiv__(self, other: Union[int, float]) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) // len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other // len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for //: "
                            f"'{type(other)}' and '{type(self)}'")
        return cn

    def __mod__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) % len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) % other)
        else:
            raise TypeError(f"unsupported operand type(s) for %: "
                            f"'{type(self)}' and '{type(other)}'")
        return cn

    def __rmod__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) % len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other % len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for %: "
                            f"'{type(other)}' and '{type(self)}'")
        return cn

    def __add__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) + len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: "
                            f"'{type(self)}' and '{type(other)}'")
        return cn

    def __radd__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) + len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) + other)
        else:
            raise TypeError(f"unsupported operand type(s) for +: "
                            f"'{type(other)}' and '{type(self)}'")
        return cn

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) * len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: "
                            f"'{type(self)}' and '{type(other)}'")
        return cn

    def __rmul__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(self) * len(other))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(len(self) * other)
        else:
            raise TypeError(f"unsupported operand type(s) for *: "
                            f"'{type(other)}' and '{type(self)}'")
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
            raise TypeError(f"unsupported operand type(s) for /: "
                            f"'{type(self)}' and '{type(other)}'")
        return cn

    def __rtruediv__(self, other: Any) -> Any:
        if isinstance(other, CircularNumber):
            cn = CircularNumber(len(other) / len(self))
        elif isinstance(other, (float, int)):
            cn = CircularNumber(other / len(self))
        else:
            raise TypeError(f"unsupported operand type(s) for /: "
                            f"'{type(other)}' and '{type(self)}'")
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
