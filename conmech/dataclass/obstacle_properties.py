from dataclasses import dataclass


@dataclass
class ObstacleProperties:
    hardness: float
    friction: float

@dataclass
class TemperatureObstacleProperties(ObstacleProperties):
    heat: float
