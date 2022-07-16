from typing import NamedTuple


class ObstacleProperties(NamedTuple):
    hardness: float
    friction: float


class TemperatureObstacleProperties(NamedTuple):
    hardness: float
    friction: float
    heat: float
