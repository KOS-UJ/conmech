import time
from dataclasses import dataclass
import pathlib
from typing import Optional


@dataclass(frozen=True)
class Config:
    """
    Technical configuration of simulations.
    """

    show: bool = True
    save: bool = False
    force: bool = True
    test: bool = False
    outputs_path: str = "./output"
    output_dir: Optional[str] = None
    shell: bool = False
    timestamp: int = time.time_ns()

    def init(self) -> "Config":
        """
        Acquires needed resources.

        Returns self so can be used as `main(Config().init())`
        """
        if self.show and self.save:
            raise ValueError("Cannot show and save at once!")

        output_dir = self.output_dir or str(self.timestamp)
        path_part_1 = pathlib.Path(self.outputs_path)
        path_part_2 = pathlib.Path(output_dir)
        path = path_part_1 / path_part_2
        path.mkdir(parents=True, exist_ok=True)
        return self
