"""
Created at 24.03.2022
"""
import pytest

from examples.example_static import main as static
from examples.example_quasistatic import main as quasistatic
from examples.example_dynamic import main as dynamic
from examples.example_temperature_dynamic import main as temperature
from examples.Jureczka_and_Ochal_2019 import main as Jureczka_and_Ochal_2019


test_suits = {
    "static": static,
    "quasistatic": quasistatic,
    "dynamic": dynamic,
    "temperature": temperature,
    "Jureczka_and_Ochal_2019": Jureczka_and_Ochal_2019,
    # TODO smoke tests for the rest examples #61
}


@pytest.fixture(params=list(test_suits.keys()))
def main(request):
    return test_suits[request.param]


def test_examples(main):
    main(show=False)
