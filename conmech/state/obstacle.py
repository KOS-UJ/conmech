from typing import Optional

import numpy as np

from conmech.properties.obstacle_properties import ObstacleProperties


class Obstacle:
    OBSTACLES = {
        "front": np.array([[[-1.0, 0.0]], [[2.0, 0.0]]]),
        "back": np.array([[[1.0, 0.0]], [[-2.0, 0.0]]]),
        "slope": np.array([[[-1.0, -2.0]], [[4.0, 0.0]]]),
        "side": np.array([[[0.0, 1.0]], [[0.0, -3.0]]]),
        "two": np.array([[[-1.0, -2.0], [-1.0, 0.0]], [[2.0, 1.0], [3.0, 0.0]]]),
        "3d": np.array([[[-1.0, -1.0, 1.0]], [[2.0, 0.0, 0.0]]]),
    }

    def __init__(self, geometry: Optional[np.ndarray], properties):
        self.geometry = geometry
        self.properties = properties

    @staticmethod
    def get_obstacle(name: str, obstacle_properties: ObstacleProperties):
        geometry = Obstacle.OBSTACLES.get(name, None)
        if geometry is None:
            raise ValueError(f"Unknown obstacle name: {name}")
        return Obstacle(geometry, obstacle_properties)
