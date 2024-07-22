# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
import numpy as np
from matplotlib import pyplot as plt

from conmech.plotting.membrane import plot_limit_points
from conmech.state.state import State
from matplotlib.ticker import StrMethodFormatter


def case1():
    states = {}
    for kappa, beta in (
        (0.0, 0.0),
        (0.0, 0.25),
        (0.0, 0.5),
        (0.0, 0.75),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (10.0, 0.5),
        (100.0, 0.5),
    )[:]:
        print((kappa, beta))
        states[kappa, beta] = State.load(
            f"output/BOSK.pub/c1_kappa={kappa:.2f};beta={beta:.2f}"
        )

    c1_reference(states, output_path="output/BOSK.pub")
    c1_steady_state(states, output_path="output/BOSK.pub")
    c1_influ(states, output_path="output/BOSK.pub")


def show(output_path, name):
    plt.gca().yaxis.set_major_formatter(
        StrMethodFormatter("{x:,.2f}")
    )  # 2 decimal places
    plt.gca().xaxis.set_major_formatter(
        StrMethodFormatter("{x:,.2f}")
    )  # 2 decimal places
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path + f"/{name}.png", format="png", dpi=800)


def c1_reference(states, output_path=None):
    plt.rc("axes", axisbelow=True)

    kappa = 100.0
    beta = 0.5
    timesteps = (0.875, 2.0)
    labels = ("b)", "a)")
    eps = 0.001

    all_zeros = []

    for subnumber, timestep in zip(labels, timesteps):
        intersect = states[kappa, beta].products["intersection at 1.00"]
        plt.figure(figsize=(5, 4))
        zeros = states[kappa, beta].products["limit points at 1.00"].data[timestep]
        all_zeros.extend(zeros)
        plt.xticks((0.0, *zeros, 1.0), rotation=90)
        plt.yticks((0.0, 1.0, 1.8))
        plt.grid()
        plt.axhspan(1.0, 1.8, alpha=0.1, color="lightskyblue")

        for t, v in intersect.data.items():
            if timestep - eps > t or t > timestep + eps:
                continue
            print(t)
            plt.plot(*v, color=f"black")
        states[kappa, beta].products["limit points at 1.00"].range(0.00, 8.00)

        plt.scatter(zeros, np.ones_like(zeros), color=f"black", s=10)
        plt.ylim(0.0, 1.8)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(rf"$\kappa={kappa:.2f}$, $\beta={beta:.2f}$, $t={timestep}$")
        plt.title(subnumber, loc="left")

        show(output_path, name=f"boundary_{timestep:.3f}")

    plt.figure(figsize=(10, 4))
    plot_limit_points(
        states[kappa, beta].products["limit points at 1.00"].range(0.00, 8.00),
        title=None,
        color=f"black",
        finish=False,
    )

    plt.xticks((0.0, *timesteps, 8.0))
    plt.yticks((0.0, *all_zeros, 1.0))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("y")
    plt.title(rf"$\kappa={kappa:.2f}$, $\beta={beta:.2f}$")
    plt.title("c)", loc="left")

    show(output_path, name="reference")


def c1_steady_state(states, output_path=None):
    plt.figure(figsize=(10, 4))
    kappa = 0.0
    beta = 0.0
    plot_limit_points(
        states[kappa, beta].products["limit points at 1.00"].range(0.00, 8.00),
        title=None,
        color=f"gray",
        finish=False,
        label=rf"$\beta={beta:.2f}$",
    )
    beta = 1.0
    plot_limit_points(
        states[kappa, beta].products["limit points at 1.00"].range(0.00, 8.00),
        title=None,
        color=f"salmon",
        finish=False,
        label=rf"$\beta={beta:.2f}$",
    )
    plt.xticks((0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.0))
    plt.yticks((0.0, 1.0))
    plt.xlabel("time")
    plt.ylabel("y")
    plt.title(rf"$\kappa={kappa:.2f}$")

    plt.legend(loc="center right")
    show(output_path, name="steady_state")


def c1_influ(states, output_path=None):
    cases = {
        ("a)", "kappa"): (
            ("lightskyblue", 100.0, 0.5),
            ("yellowgreen", 10.0, 0.5),
            ("gold", 1.0, 0.5),
            ("salmon", 0.0, 0.5),
        ),
        ("b)", "beta"): (
            ("lightskyblue", 0.0, 0.25),
            ("yellowgreen", 0.0, 0.5),
            ("gold", 0.0, 0.75),
            ("salmon", 0.0, 1.0),
        ),
    }
    for (subnumber, var), case in cases.items():
        plt.figure(figsize=(6, 4.5))
        for c, kappa, beta in case:
            print((kappa, beta))
            variable = kappa if var == "kappa" else beta
            plot_limit_points(
                states[kappa, beta].products["limit points at 1.00"].range(0.00, 4.00),
                title=None,
                label=rf"$\{var}={variable:.2f}$",
                finish=False,
                color=f"{c}",
            )

        plt.legend(loc="center right")
        plt.xticks((0.0, 0.92, 1.8, 2.65, 3.6, 4.0))
        plt.yticks((0.0, 0.5, 1.0))
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("y")
        const_name = "kappa" if var == "beta" else "beta"
        const_value = kappa if var == "beta" else beta
        plt.title(rf"$\{const_name}={const_value:.2f}$")
        plt.title(subnumber, loc="left")

        show(output_path, name="var_" + var)


def case2():
    output_path = "output/BOSK.pub"
    kappa = 1.0
    # for var, subnumber, beta in (('control', 'a)', 0.), ('beta', 'b)', 100.)):
    # plt.figure(figsize=(6, 4.5))
    # state = State.load(f"output/BOSK.pub/i2_kappa={kappa:.2f};beta={beta:.2f}")
    # plot_limit_points(
    #     state.products['limit points at 0.50'],
    #     title=fr'$\kappa={kappa}$ $\beta={beta}$', finish=False)
    # state = State.load(f"output/BOSK.pub/ci2_kappa={kappa:.2f};beta={beta:.2f}")
    # plot_limit_points(
    #     state.products['limit points at 0.50'],
    #     title=fr'$\kappa={kappa}$ $\beta={beta}$', finish=False)
    # plt.title(subnumber, loc='left')
    # plt.xticks((0.0, 3.6, 7.2, 10.8, 14.4, 16.0))
    # plt.yticks((0.0, 0.5, 1.0))
    # plt.xlabel("time")
    # plt.ylabel("y")
    # plt.grid()
    #
    # show(output_path, name='int_' + var)

    beta = 0.0  # TODO
    state = State.load(f"output/BOSK.pub/i2_c2_plain")
    plot_limit_points(
        state.products["limit points at 0.50"].range(0.0, 4.0),
        title=rf"$\kappa={kappa}$ $\beta={beta}$",
        finish=False,
    )
    plt.show()

    timesteps = (
        1.0 - 0.0625,
        1.75,
    )
    labels = ("b)", "a)")
    eps = 0.001

    all_zeros = []

    for subnumber, timestep in zip(labels, timesteps):
        intersect = state.products["intersection at 0.50"]
        plt.figure(figsize=(5, 4))
        zeros = state.products["limit points at 0.50"].data[timestep]
        all_zeros.extend(zeros)
        plt.xticks((0.0, *zeros, 1.0), rotation=90)
        plt.yticks((0.0, 1.0, 1.8))
        plt.grid()
        plt.axhspan(1.0, 1.8, alpha=0.1, color="lightskyblue")

        for t, v in intersect.data.items():
            if timestep - eps > t or t > timestep + eps:
                continue
            print(t)
            plt.plot(*v, color=f"black")
        state.products["limit points at 0.50"].range(0.00, 8.00)

        plt.scatter(zeros, np.ones_like(zeros), color=f"black", s=10)
        plt.ylim(0.0, 1.8)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(rf"$\kappa={kappa:.2f}$, $\beta={beta:.2f}$, $t={timestep}$")
        plt.title(subnumber, loc="left")

        plt.show()

        # show(output_path, name=f'boundary_{timestep:.3f}')


if __name__ == "__main__":
    # case1()
    case2()
