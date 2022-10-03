import time
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Config:
    shell: bool = False
    timestamp_skip: int = 10000
    run_timestamp: float = int(time.time() * timestamp_skip)
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")

    print_skip: float = 0.1 #0.01

    plot_tests: bool = False

    output_catalog: str = "output"
