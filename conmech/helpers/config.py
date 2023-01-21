import os
import time
from dataclasses import dataclass
from datetime import datetime

# def set_env(test=False):
#     if "ENV_READY" not in os.environ or not os.environ["ENV_READY"]:
#         return
#     if not test:
#         # os.environ["JAX_ENABLE_X64"] = "1"
#         os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#         # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "-1"
#         # os.environ["JAX_PLATFORM_NAME"] = "cpu"
#         # os.environ["JAX_DISABLE_JIT"] = "1"
#         # os.environ["JAX_DEBUG_NANS"] = "1"

#         # os.environ['OPTIMIZATION_BACKEND'] = "cpu" # "gpu" "cpu" None

#         # import lovely_jax as lj
#         # import lovely_tensors as lt
#         # lt.monkey_patch()
#         # lj.monkey_patch()
#     else:
#         os.environ["JAX_ENABLE_X64"] = "0"
#         os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#         os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#         os.environ["JAX_PLATFORM_NAME"] = "cpu"
#         os.environ["JAX_DISABLE_JIT"] = "0"
#         os.environ["JAX_DEBUG_NANS"] = "0"

#         os.environ["OPTIMIZATION_BACKEND"] = "cpu"

#     os.environ["ENV_READY"] = "1"


@dataclass
class SimulationConfig:
    use_normalization: bool = False
    use_linear_solver: bool = False
    use_green_strain: bool = True
    use_nonconvex_friction_law: bool = False
    use_constant_contact_integral: bool = False
    use_lhs_preconditioner: bool = False
    use_pca: bool = False


@dataclass
class Config:
    shell: bool = False
    timestamp_skip: int = 10000
    run_timestamp: float = int(time.time() * timestamp_skip)
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")
    verbose: bool = True

    animation_backend: str = "matplotlib blender"  # blender matplotlib
    blender_output: bool = True
    print_skip: float = 0.1  # 0.01
    plot_tests: bool = False
    output_catalog: str = "output"
