# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2023  Piotr Bartman <piotr.bartman@uj.edu.pl>
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
import pickle
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


from conmech.helpers.config import Config
from conmech.mesh.boundaries_description import BoundariesDescription
from conmech.plotting.drawer import Drawer
from conmech.scenarios.problems import RelaxationQuasistaticProblem
from conmech.simulations.problem_solver import QuasistaticRelaxation

from examples.p_slope_contact_law import make_slope_contact_law

eps = 1e-18
ox = 2.5
oy = 2.0
r_big = 2.5
r_small = 1.5
r = (r_big + r_small) / 2
fv = 0.1
TEST = 1


@dataclass
class QuasistaticSetup(RelaxationQuasistaticProblem):
    grid_height: ... = 1.0
    elements_number: ... = (20 / TEST, 20 / TEST)
    mu_coef: ... = 714.29
    la_coef: ... = 2857.14
    time_step: ... = 1/128 * TEST**2
    contact_law: ... = make_slope_contact_law(slope=27)

    @staticmethod
    def relaxation(t: float) -> np.ndarray:
        _mu = 0.
        return np.array(
            [[[2 * _mu, 0], [_mu, _mu]],
             [[_mu, _mu], [0, 2 * _mu]], ]
        )

    @staticmethod
    def inner_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def outer_forces(x, t=None):
        return np.array([0.0, 0.0])

    @staticmethod
    def friction_bound(u_nu):
        return 0

    boundaries: ... = BoundariesDescription(
        contact=lambda x: x[0] >= 4 and x[1] < eps, dirichlet=lambda x: x[0] <= 1 and x[1] < eps
    )


def main(save: bool = False, simulate: bool = True):
    """

    If not `simulate` but results is not ready: run simulation.
    If `simulate` simulation will be always run.

    If `save` results will be saved as file.
    If not `save` results will be shown.
    """
    config = Config()
    setup = QuasistaticSetup(mesh_type="tunnel")
    h = setup.elements_number[0]

    def sin_outer_forces(x, t=None):
        if x[1] <= oy:
            return np.array([0., 0])
        if (x[0] - ox) ** 2 + (x[1] - oy) ** 2 >= (r + eps) ** 2:
            return np.array([0, fv * t * np.sin(t)])
        return np.array([0.0, 0.0])

    def const_outer_forces(x, t=None):
        if x[1] <= oy:
            return np.array([0., 0])
        if (x[0] - ox) ** 2 + (x[1] - oy) ** 2 >= (r + eps) ** 2:
            return np.array([0, -0.5])
        return np.array([0.0, 0.0])

    def const_relaxation(t=None):
        _mu = 1000.
        return np.array(
            [[[2 * _mu, 0], [_mu, _mu]],
             [[_mu, _mu], [0, 2 * _mu]], ]
        )

    def linear_relaxation(t=None):
        if t < 1.5:
            _mu = 1000. * t
        elif t < 3.0:
            _mu = 1000. * (3.0 - t)
        else:
            _mu = 0.
        return np.array(
            [[[2 * _mu, 0], [_mu, _mu]],
             [[_mu, _mu], [0, 2 * _mu]], ]
        )

    examples = {
        "sob_01": {
            "n_steps": 512,
            "output_steps": (0, 192, 416, 512),
            "outer_forces": sin_outer_forces,
            "relaxation": const_relaxation,
        },
        "sob_02": {
            "n_steps": 512,
            "output_steps": range(0, 512, 16),
            "outer_forces": const_outer_forces,
            "relaxation": linear_relaxation,
        }
    }

    try:
        for name in examples.keys():
            steps = examples[name]["output_steps"]
            with open(f"./output/sob2023/{name}_h_{h}_global", "rb") as output:
                _ = pickle.load(output)

            with open(f"./output/sob2023/{name}_h_{h}_penetration", "rb") as output:
                _ = pickle.load(output)

            for time_step in steps:
                with open(f"./output/sob2023/{name}_t_{time_step}_h_{h}", "rb") as output:
                    _ = pickle.load(output)
    except IOError:
        simulate = True
    except EOFError:
        simulate = True

    if simulate:
        for name in examples.keys():
            setup.outer_forces = examples[name]["outer_forces"]
            setup.relaxation = examples[name]["relaxation"]

            runner = QuasistaticRelaxation(setup, solving_method="schur")

            states = runner.solve(
                n_steps=examples[name]["n_steps"],
                output_step=examples[name]["output_steps"],
                verbose=False,
                initial_absement=setup.initial_absement,
                initial_displacement=setup.initial_displacement,
            )
            f_max = -np.inf
            f_min = np.inf
            for state in states:
                f_max = max(f_max, np.max(state.stress_x))
                f_min = min(f_min, np.min(state.stress_x))
                with open(
                        f"./output/sob2023/{name}_t_{int(state.time // setup.time_step)}_h_{h}",
                        "wb+",
                ) as output:
                    # Workaround
                    relaxation = state.body.body_prop.relaxation
                    state.body.outer_forces = None
                    state.body.inner_forces = None
                    state.body.body_prop.relaxation = None
                    pickle.dump(state, output)
                    state.body.body_prop.relaxation = relaxation
            with open(
                    f"./output/sob2023/{name}_h_{h}_penetration",
                    "wb+",
            ) as output:
                pickle.dump(runner.penetration, output)
            with open(
                    f"./output/sob2023/{name}_h_{h}_global",
                    "wb+",
            ) as output:
                pickle.dump([f_min, f_max], output)

    for name in examples.keys():
        steps = examples[name]["output_steps"]
        with open(f"./output/sob2023/{name}_h_{h}_global", "rb") as output:
            f_limits = pickle.load(output)
        with open(f"./output/sob2023/{name}_h_{h}_penetration", "rb") as output:
            pnt = np.asarray(pickle.load(output))
        fig, axes = plt.subplots(3, 1)
        t = np.asarray(range(0, examples[name]["n_steps"] + 1)) * setup.time_step
        frc = np.empty((examples[name]["n_steps"] + 1, 1))
        for i, _t in enumerate(t):
            frc[i, :] = examples[name]["outer_forces"](np.asarray([2.5, 4.5]), _t)[1]
        rlx = np.empty((examples[name]["n_steps"] + 1, 1))
        for i, _t in enumerate(t):
            rlx[i, :] = examples[name]["relaxation"](_t)[1, 0, 1]
        for ax in axes:
            ax.set_xlim(0.0, 4.0)
        axes[0].set_ylim(-0.4, 0.4)
        axes[1].set_ylim(-100, 1600)
        axes[2].set_ylim(-0.1, 0.1)
        axes[0].plot(t, frc, color="black")

        axes[0].axhline(y=[0], color='gray', ls='-', lw=1)
        axes[1].axhline(y=[0], color='gray', ls='-', lw=1)
        axes[2].axhline(y=[0], color='gray', ls='-', lw=1)
        axes[1].plot(t, rlx, color="black")
        pnt_sig_change = 0
        old_p = pnt[0, 1]
        for t, p in pnt[1:, :]:
            if old_p > 0 and p <= 0:
                pnt_sig_change = t
                break
            old_p = p
        axes[0].axvline(x=[pnt_sig_change], color='gray', ls='-', lw=1)
        axes[1].axvline(x=[pnt_sig_change], color='gray', ls='-', lw=1)
        axes[2].axvline(x=[pnt_sig_change], color='gray', ls='-', lw=1)
        axes[2].plot(pnt[:, 0], pnt[:, 1], color="black")
        plt.show()

        for time_step in steps:
            with open(f"./output/sob2023/{name}_t_{time_step}_h_{h}", "rb") as output:
                state = pickle.load(output)
                # Workaround
                state.body.outer_forces = examples[name]["outer_forces"]
                state.body.body_prop.relaxation = examples[name]["relaxation"]

                drawer = Drawer(state=state, config=config)
                drawer.node_size = 0
                drawer.original_mesh_color = None
                drawer.deformed_mesh_color = "black"
                drawer.normal_stress_scale = 1
                drawer.field_name = None
                drawer.xlabel = "x"
                drawer.ylabel = "y"

                if time_step == 0:
                    fig, axes = plt.subplots(1, 1)
                    axes = (axes,)
                    axes[0].annotate('$\Gamma_1$', xy=(0, 0), xytext=(0.33, -0.50), fontsize=18)
                    axes[0].annotate('$\Gamma_2$', xy=(0, 0), xytext=(4.33, 5.00), fontsize=18)
                    axes[0].annotate('$\Gamma_2$', xy=(0, 0), xytext=(2.33, 3.0), fontsize=18)
                    axes[0].annotate('$\mathbf{f}_2$', xy=(0, 0), xytext=(2.5, 5.00), fontsize=15)
                    axes[0].annotate('$\Gamma_3$', xy=(0, 0), xytext=(4.33, -0.50), fontsize=18)
                    drawer.outer_forces_scale = -1
                    plt.title("Reference configuration")
                    # to have nonzero force interface on Neumann boundary.
                    state.time = 4
                else:
                    drawer.outer_forces_scale = -1 # TODO
                    fig, axes = plt.subplots(1, 2)
                    drawer.x_min = 3.4
                    drawer.x_max = 5.6
                    drawer.y_min = -1
                    drawer.y_max = 1.5
                    drawer.ylabel = None
                    drawer.draw(
                        fig_axes=(fig, axes[1]),
                        show=False,
                        field_min=f_limits[0],
                        field_max=f_limits[1],
                        save=False,
                    )
                    drawer.ylabel = "y"
                    axes[1].axis("on")
                    axes[1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                    axes[1].set_aspect("equal", adjustable="box")
                    zoom_outside(axes[0], [3, -1.5, 6, 2], axes[1], color="gray")

                    fig.suptitle(f"time: {state.time:.2f}")
                drawer.x_min = 0
                drawer.x_max = 5.0
                drawer.y_min = -0.9
                drawer.y_max = 4.9
                drawer.draw(
                    fig_axes=(fig, axes[0]),
                    show=False,
                    field_min=f_limits[0],
                    field_max=f_limits[1],
                    save=False,
                )
                axes[0].axis("on")
                axes[0].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
                axes[0].set_aspect("equal", adjustable="box")

                fig.tight_layout(rect=[0, 0, 1, 1.2])
                plt.subplots_adjust(wspace=0.4, top=1.25)
                if not save:
                    plt.show()
                if save:
                    drawer.save_plot("pdf", name=f"{name}_{time_step}")


def zoom_outside(
        src_ax,
        roi,
        dst_ax,
        color="red",
        linewidth=2,
        draw_lines=False,
        roi_kwargs=None,
        arrow_kwargs=None,
):
    """Create a zoomed subplot outside the original subplot

    src_ax: matplotlib.axes
        Source axis where locates the original chart
    dst_ax: matplotlib.axes
        Destination axis in which the zoomed chart will be plotted
    roi: list
        Region Of Interest is a rectangle defined by [xmin, ymin, xmax, ymax],
        all coordinates are expressed in the coordinate system of data
    roiKwargs: dict (optional)
        Properties for matplotlib.patches.Rectangle given by keywords
    arrowKwargs: dict (optional)
        Properties used to draw a FancyArrowPatch arrow in annotation
    """
    roi_kwargs = roi_kwargs if roi_kwargs else {}
    arrow_kwargs = arrow_kwargs if arrow_kwargs else {}
    roi_kwargs = dict(
        [("fill", False), ("linestyle", "dashed"), ("color", color), ("linewidth", linewidth)]
        + list(roi_kwargs.items())
    )
    arrow_kwargs = dict(
        [("arrowstyle", "-"), ("color", color), ("linewidth", linewidth)]
        + list(arrow_kwargs.items())
    )
    # draw a rectangle on original chart
    src_ax.add_patch(Rectangle([roi[0], roi[1]], roi[2] - roi[0], roi[3] - roi[1], **roi_kwargs))

    if not draw_lines:
        return

    # get coordinates of corners
    src_corners = [[roi[0], roi[1]], [roi[0], roi[3]], [roi[2], roi[1]], [roi[2], roi[3]]]
    dst_corners = dst_ax.get_position().corners()
    src_bb = src_ax.get_position()
    dst_bb = dst_ax.get_position()
    # find corners to be linked
    if src_bb.max[0] <= dst_bb.min[0]:  # right side
        if src_bb.min[1] < dst_bb.min[1]:  # upper
            corners = [1, 2]
        elif src_bb.min[1] == dst_bb.min[1]:  # middle
            corners = [0, 1]
        else:
            corners = [0, 3]  # lower
    elif src_bb.min[0] >= dst_bb.max[0]:  # left side
        if src_bb.min[1] < dst_bb.min[1]:  # upper
            corners = [0, 3]
        elif src_bb.min[1] == dst_bb.min[1]:  # middle
            corners = [2, 3]
        else:
            corners = [1, 2]  # lower
    elif src_bb.min[0] == dst_bb.min[0]:  # top side or bottom side
        if src_bb.min[1] < dst_bb.min[1]:  # upper
            corners = [0, 2]
        else:
            corners = [1, 3]  # lower
    else:
        RuntimeWarning("Cannot find a proper way to link the original chart to "
                       "the zoomed chart! The lines between the region of "
                       "interest and the zoomed chart will not be plotted.")
        return
    # plot 2 lines to link the region of interest and the zoomed chart
    for k in range(2):
        src_ax.annotate('', xy=src_corners[corners[k]], xycoords="data",
                        xytext=dst_corners[corners[k]], textcoords="figure fraction",
                        arrowprops=arrow_kwargs)


if __name__ == "__main__":
    main(simulate=True, save=True)
