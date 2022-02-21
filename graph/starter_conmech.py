# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
from sympy import false
from torchaudio import datasets

import config
import examples
import drawer
import printer
from calculator import Calculator
from drawer import Drawer
from mesh import Mesh
from mesh_features import MeshFeatures
from setting import *
import mapper
import resource


def run_conmech_dynamic(
    all_forces_functions, simulate_dirty_data=True
):
    timestamp = helpers.get_timestamp()
    for forces_function in all_forces_functions:
        path = f"{timestamp} - CONMECH"
        if simulate_dirty_data:
            path += " DIRTY"

        printer.print_one_dynamic(
            lambda setting : Calculator(setting).solve_function() ,
            forces_function,
            path,
            simulate_dirty_data=simulate_dirty_data,
            print_base=False,
            print_max_data=True,
            description=f"Printing conmech {forces_function.__name__}"
        )


def main():
    print_forces_functions = [examples.f_obstacle]
    run_conmech_dynamic(
       print_forces_functions, simulate_dirty_data=False
    )

if __name__ == "__main__":
    main()
