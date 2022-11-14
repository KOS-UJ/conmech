"""
Created at 24.03.2022
"""
import shutil

import pytest

from conmech.helpers.config import Config
from examples.example_static import main as static
from examples.example_quasistatic import main as quasistatic
from examples.example_piezo_quasistatic import main as quasi_piezo
from examples.example_dynamic import main as dynamic
from examples.example_piezoelectric_dynamic import main as dynamic_piezo
from examples.example_temperature_dynamic import main as dynamic_temp
from examples.examples_2d import main as examples_2d
from examples.examples_3d import main as examples_3d
from examples.examples_temperature_2d import main as examples_temperature_2d
from examples.examples_temperature_3d import main as examples_temperature_3d
from examples.Jureczka_and_Ochal_2019 import main as Jureczka_and_Ochal_2019
from examples.Jureczka_Ochal_Bartman_2023 import main as Jureczka_Ochal_Bartman_2023
from examples.Sofonea_Ochal_Bartman_2023 import main as Sofonea_Ochal_Bartman_2023
from examples.examples_poisson import main as poisson

default_args = dict(show=False, save=False, force=True, test=True)
default_args_deep = dict(mesh_density=4, final_time=0.05, plot_animation=False)

test_suits = {
    "poisson": lambda: poisson(**default_args),
    "static": lambda: static(Config(**default_args).init()),
    "quasistatic": lambda: quasistatic(Config(**default_args).init()),
    "quasi_piezo": lambda: quasi_piezo(Config(**default_args).init()),
    "dynamic": lambda: dynamic(Config(**default_args).init()),
    "dynamic_piezo": lambda: dynamic_piezo(Config(**default_args).init()),
    "dynamic_temp": lambda: dynamic_temp(Config(**default_args).init()),
    "Jureczka_and_Ochal_2019": lambda: Jureczka_and_Ochal_2019(Config(**default_args).init()),
    "Jureczka_Ochal_Bartman_2023": lambda: Jureczka_Ochal_Bartman_2023(
        Config(outputs_path="./output/JOB2023", **default_args).init()
    ),
    "Sofonea_Ochal_Bartman_2023": lambda: Sofonea_Ochal_Bartman_2023(
        Config(outputs_path="./output/SOB2023", **default_args).init()
    ),
    "examples_2d": lambda: examples_2d(Config(**default_args).init(), **default_args_deep),
    "examples_3d": lambda: examples_3d(Config(**default_args).init(), **default_args_deep),
    "examples_temperature_2d": lambda: examples_temperature_2d(
        Config(**default_args).init(), **default_args_deep
    ),
    "examples_temperature_3d": lambda: examples_temperature_3d(
        Config(**default_args).init(), **default_args_deep
    ),
}


@pytest.fixture(params=list(test_suits.keys()))
def main_function(request):
    return test_suits[request.param]


def test_examples(main_function):
    main_function()
    shutil.rmtree("./output")
