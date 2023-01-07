import os
import time
from dataclasses import dataclass
from datetime import datetime

NORMALIZE = False
USE_LINEAR_SOLVER = False
USE_GREEN_STRAIN = True
USE_NONCONVEX_FRICTION_LAW = False
USE_CONSTANT_CONTACT_INTEGRAL = False

OPTIMIZATION_BACKEND = None  # "gpu" "cpu" None


def set_jax():
    jax_64 = False

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "-1"
    # os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # os.environ["JAX_DISABLE_JIT"] = "1"

    # os.environ['OPTIMIZATION_BACKEND'] = "cpu"

    if jax_64:
        os.environ["JAX_ENABLE_X64"] = "1"
        print("JAX 64 BIT MODE")
    else:
        print("JAX 32 BIT MODE")


@dataclass
class Config:
    shell: bool = False
    timestamp_skip: int = 10000
    run_timestamp: float = int(time.time() * timestamp_skip)
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")

    print_skip: float = 0.1  # 0.01

    plot_tests: bool = False

    output_catalog: str = "output"
