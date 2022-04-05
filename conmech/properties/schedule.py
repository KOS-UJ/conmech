from dataclasses import dataclass


@dataclass
class Schedule:
    final_time: float
    time_step: float = 0.01

    @property
    def episode_steps(self):
        return int(self.final_time / self.time_step)
