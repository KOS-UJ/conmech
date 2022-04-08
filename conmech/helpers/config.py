import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:
    SHELL: bool = False
    TIMESTAMP_SKIP = 10000
    RUN_TIMESTAMP: float = int(time.time() * TIMESTAMP_SKIP)
    CURRENT_TIME: str = datetime.now().strftime("%m.%d-%H.%M.%S")

    NORMALIZE_ROTATE = True

    PRINT_SKIP = 0.1

    PLOT_TESTS = False
