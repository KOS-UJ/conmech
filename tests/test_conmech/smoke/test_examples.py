"""
Created at 24.03.2022
"""
import numpy as np
import pytest

from examples.example_dynamic import main as dynamic
from examples.example_piezoelectric_dynamic import main as piezoelectric_dynamic
from examples.example_piezoelectric_quasistatic import main as piezoelectric_quasistatic
from examples.example_quasistatic import main as quasistatic
from examples.example_static import main as static
from examples.example_temperature_dynamic import main as temperature
from examples.examples_2d import main as examples_2d
from examples.examples_3d import main as examples_3d
from examples.examples_temperature_2d import main as examples_temperature_2d
from examples.examples_temperature_3d import main as examples_temperature_3d
from examples.Jureczka_and_Ochal_2019 import main as Jureczka_and_Ochal_2019
from examples.temperature_2023 import main as temperature_2023

default_args = dict(show=False, save=False)
default_args_deep = dict(mesh_density=4, final_time=0.05, plot_animation=False)
default_args_temp = dict(mesh_density=8, final_time=0.02, plot_animation=False)


examples_2d_samples = {
    "displacement_1": np.array(
        [
            [6.00356515e-03, -7.06660383e-06],
            [6.00540195e-03, -1.43744906e-06],
            [5.99914029e-03, -3.08102671e-06],
            [5.99963129e-03, -4.40203552e-06],
            [6.00377824e-03, -3.00577191e-06],
        ]
    ),
    "displacement_3": np.array(
        [
            [2.81884125e-03, -1.17093241e-03],
            [3.08342716e-03, -6.23409182e-04],
            [1.22191360e-03, 1.07393786e-05],
            [2.98239775e-03, -1.02297147e-03],
            [3.05907561e-03, -9.11544901e-04],
        ]
    ),
}

examples_3d_samples = {
    "displacement_1": np.array(
        [
            [6.70247002e-06, -1.85000553e-05, -1.47760304e-03],
            [9.05754918e-06, -1.97138080e-05, -1.47420336e-03],
            [5.17022712e-06, -1.93989933e-05, -1.47767968e-03],
            [8.04305741e-07, -2.09521866e-05, -1.47756168e-03],
            [1.09347499e-05, -1.16621378e-05, -1.47506782e-03],
        ]
    ),
    "displacement_3": np.array(
        [
            [3.39022368e-04, 3.53401051e-04, 3.53442993e-04],
            [3.09904231e-04, 8.85665204e-06, 8.91434552e-06],
            [3.23197141e-04, -3.33290646e-04, -3.33249200e-04],
            [8.71500289e-04, 3.39559484e-04, 3.63094708e-04],
            [8.65830332e-04, 4.81930430e-06, 6.24605387e-06],
        ]
    ),
}


examples_temperature_2d_samples = {
    "displacement_1": np.array(
        [
            [0.00142149, -0.0009216],
            [0.00151364, -0.00071688],
            [0.00040353, -0.00034748],
            [0.00148845, -0.0008388],
            [0.00150865, -0.00079168],
        ]
    ),
    "temperature_1": np.array(
        [
            [-2.56697618e-04],
            [1.93818123e-05],
            [4.25329781e-04],
            [-1.06643965e-04],
            [-2.62127269e-05],
        ]
    ),
    "displacement_3": np.array(
        [
            [0.00128513, 0.00076893],
            [0.00090915, -0.00049011],
            [0.0034406, -0.00068195],
            [0.0025061, 0.00064361],
            [0.00117007, 0.0003193],
        ]
    ),
    "temperature_3": np.array(
        [[1.2393272e-05], [6.2217747e-05], [-5.6820531e-06], [-1.2859185e-05], [8.9635440e-05]]
    ),
}


examples_temperature_3d_samples = {
    "displacement_1": np.array(
        [
            [0.00103484, 0.00013086, 0.00013599],
            [0.001085, 0.00011871, 0.00012759],
            [0.00112512, 0.00011901, 0.00013564],
            [0.00107998, 0.00012467, 0.00013759],
            [0.00113015, 0.00011134, 0.0001251],
        ]
    ),
    "temperature_1": np.array(
        [[7.0537436e-07], [2.2259915e-06], [8.3138127e-07], [1.2094683e-06], [1.3304393e-06]]
    ),
    "displacement_3": np.array(
        [
            [6.33643693e-06, -7.09279717e-06, -5.94471879e-04],
            [7.25792997e-06, -4.18800555e-06, -5.95096145e-04],
            [6.29875693e-06, -5.01875824e-06, -5.95976226e-04],
            [5.28236589e-06, -6.97164724e-06, -5.95790334e-04],
            [7.34179834e-06, -1.52833341e-06, -5.95447011e-04],
        ]
    ),
    "temperature_3": np.array(
        [[4.3151076e-08], [2.1226799e-07], [1.0750851e-07], [3.8287749e-07], [3.1002946e-07]]
    ),
}

temperature_2023_samples = {
    "displacement_1": np.array(
        [
            [-2.06824676e-04, -1.64975652e-06, -5.94950411e-04],
            [-2.06267764e-04, 7.74510554e-07, -5.95232081e-04],
            [-2.06869529e-04, 8.45852919e-07, -5.97352143e-04],
            [-2.07442380e-04, -1.19468339e-06, -5.96809573e-04],
            [-2.06354377e-04, 3.63706291e-06, -5.95994741e-04],
        ]
    ),
    "temperature_1": np.array(
        [[-2.8604450e-08], [-1.4810604e-07], [3.1039036e-08], [2.8930788e-07], [5.0962576e-08]]
    ),
    "displacement_3": np.array(
        [
            [-2.02650749e-04, -1.58078317e-05, -5.90169467e-04],
            [-1.99464797e-04, -1.43445618e-05, -5.90849426e-04],
            [-2.03456143e-04, -1.46893872e-05, -5.92570212e-04],
            [-2.05477201e-04, -1.54257572e-05, -5.92287648e-04],
            [-1.97564326e-04, -1.19456157e-05, -5.91684338e-04],
        ]
    ),
    "temperature_3": np.array(
        [[4.2536016e-07], [7.1196030e-07], [3.4091278e-07], [1.2080847e-06], [6.5076665e-07]]
    ),
}


test_suits = {
    "static": (lambda: static(**default_args), None),
    "quasistatic": (lambda: quasistatic(**default_args), None),
    "dynamic": (lambda: dynamic(**default_args), None),
    "temperature": (lambda: temperature(**default_args), None),
    "piezo_quasistatic": (lambda: piezoelectric_quasistatic(**default_args), None),
    "piezoelectric_dynamic": (lambda: piezoelectric_dynamic(**default_args), None),
    "Jureczka_and_Ochal_2019": (lambda: Jureczka_and_Ochal_2019(**default_args), None),
    "examples_2d": (lambda: examples_2d(**default_args_deep), examples_2d_samples),
    "examples_3d": (lambda: examples_3d(**default_args_deep), examples_3d_samples),
    "examples_temperature_2d": (
        lambda: examples_temperature_2d(**default_args_deep),
        examples_temperature_2d_samples,
    ),
    "examples_temperature_3d": (
        lambda: examples_temperature_3d(**default_args_temp),
        examples_temperature_3d_samples,
    ),
    "temperature_2023": (
        lambda: temperature_2023(**default_args_temp),
        temperature_2023_samples,
    ),
}


@pytest.fixture(params=list(test_suits.keys()))
def example_args(request):
    return test_suits[request.param]


def test_examples(example_args):
    main_function, expected_samples = example_args
    scenes = main_function()

    if expected_samples is None:
        print("Skipping testing of results")
        return

    def compare_values(actual, expected):
        np.testing.assert_array_almost_equal(actual, expected, decimal=3)

    for i, scene in enumerate(scenes):
        key = f"displacement_{i+1}"
        if key in expected_samples:
            print("Testing displacement")
            expected_sample = expected_samples[key]
            actual_sample = scene.displacement_old[: len(expected_sample)]
            compare_values(actual_sample, expected_sample)
        key = f"temperature_{i+1}"
        if key in expected_samples:
            print("Testing temperature")
            expected_sample = expected_samples[key]
            actual_sample = scene.t_old[: len(expected_sample)]
            compare_values(actual_sample, expected_sample)
