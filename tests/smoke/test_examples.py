"""
Created at 24.03.2022
"""
import pytest

from examples.example_static import main as static
from examples.example_quasistatic import main as quasistatic
from examples.example_dynamic import main as dynamic
from examples.example_temperature_dynamic import main as temperature
from examples.Jureczka_and_Ochal_2019 import main as Jureczka_and_Ochal_2019
from examples.examples_graph import main as examples_graph
from examples.examples_roll import main as examples_roll
from examples.examples_temperature import main as examples_temperature
from examples.examples_3d import main as examples_3d


default_args = dict(show=False)
default_args_deep = dict(mesh_density=2, final_time=0.5, plot_animation=False)

test_suits = {
    "static": lambda: static(**default_args),
    "quasistatic": lambda: quasistatic(**default_args),
    "dynamic": lambda: dynamic(**default_args),
    "temperature": lambda: temperature(**default_args),
    "Jureczka_and_Ochal_2019": lambda: Jureczka_and_Ochal_2019(**default_args),

    "examples_graph": lambda: examples_graph(**default_args_deep),
    "examples_roll": lambda: examples_roll(**default_args_deep),
    "examples_temperature": lambda: examples_temperature(**default_args_deep),
    "examples_3d": lambda: examples_3d(**default_args_deep)
}

@pytest.fixture(params=list(test_suits.keys()))
def main_function(request):
    return test_suits[request.param]

def test_examples(main_function):
    main_function()
