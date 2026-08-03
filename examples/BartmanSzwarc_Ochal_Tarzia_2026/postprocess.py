# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2025-2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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

import gc
import string
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import griddata

from conmech.state.state import TemperatureState

from examples.BartmanSzwarc_Ochal_Tarzia_2026.run import load_or_simulate


def _pair_label(alpha, ih) -> str:
    result = "("
    if alpha == np.inf:
        result += "∞"
    else:
        result += f"{alpha:.2E}"
    result += ", "
    result += f"1/{ih})"
    return result


def _relative_temperature_error(
    reference_state: TemperatureState, state: TemperatureState
) -> float:
    # Project `state` temperature onto reference mesh nodes and compute
    # the (absolute) L2 norm of the difference using element mass matrices
    ref_nodes = np.asarray(reference_state.body.mesh.nodes, dtype=float)
    ref_values = np.asarray(reference_state.temperature, dtype=float).ravel()
    nodes = np.asarray(state.body.mesh.nodes, dtype=float)
    values = np.asarray(state.temperature, dtype=float).ravel()

    # interpolate values from `state` nodes to `reference_state` nodes
    interpolated = np.asarray(
        griddata(nodes, values, ref_nodes, method="linear"), dtype=float
    )  # type: ignore[arg-type]
    if np.isnan(interpolated).any():
        nearest = np.asarray(
            griddata(nodes, values, ref_nodes, method="nearest"), dtype=float
        )  # type: ignore[arg-type]
        interpolated = np.where(np.isnan(interpolated), nearest, interpolated)

    diff_at_nodes = ref_values - interpolated

    # If mesh has elements, compute element-wise integral using P1 mass matrix
    try:
        tris = np.asarray(reference_state.body.mesh.elements, dtype=int)
    except Exception:
        tris = np.asarray([], dtype=int)

    if tris.size == 0:
        # Fallback: simple discrete L2 on nodes (not weighted by area)
        return float(np.sqrt(np.sum(diff_at_nodes * diff_at_nodes)))

    l2_sq = 0.0
    coords = ref_nodes
    for tri in tris:
        i, j, k = tri[0], tri[1], tri[2]
        xi = coords[i]
        xj = coords[j]
        xk = coords[k]
        # triangle area
        area = 0.5 * abs((xj[0] - xi[0]) * (xk[1] - xi[1]) - (xk[0] - xi[0]) * (xj[1] - xi[1]))
        # local mass matrix for linear triangle: (area/12) * [[2,1,1],[1,2,1],[1,1,2]]
        uloc = np.array([diff_at_nodes[i], diff_at_nodes[j], diff_at_nodes[k]], dtype=float)
        # compute uloc^T M uloc
        mloc_factor = area / 12.0
        # exact computation: 2*u0^2 + 2*u1^2 + 2*u2^2 + 2*(u0*u1 + u0*u2 + u1*u2)
        a0, a1, a2 = uloc[0], uloc[1], uloc[2]
        contrib = mloc_factor * (
            2.0 * (a0 * a0 + a1 * a1 + a2 * a2) + 2.0 * (a0 * a1 + a0 * a2 + a1 * a2)
        )
        l2_sq += contrib

    return float(np.sqrt(max(l2_sq, 0.0)))


def draw_temperature_grid(config, to_plot):
    rows = len(to_plot)
    cols = len(to_plot[0])
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    # Do a single-pass scan to compute global field min/max without keeping states in memory.
    field_min = float("inf")
    field_max = float("-inf")
    for row in to_plot:
        for alpha, ih in row:
            print(f"Scanning {alpha=}, {ih=} for field range")
            state = load_or_simulate(config, alpha, ih)
            assert state is not None
            try:
                temperature = np.asarray(state.temperature, dtype=float).ravel()
                field_min = min(field_min, float(temperature.min().item()))
                field_max = max(field_max, float(temperature.max().item()))
            finally:
                # release heavy object before next iteration
                del state
                gc.collect()

    # Now render each subplot, loading and discarding state one-by-one to limit RAM usage
    seq_num = 0
    for row_idx, row in enumerate(to_plot):
        for col_idx, (alpha, ih) in enumerate(row):
            print(f"Rendering {alpha=}, {ih=}")
            state = load_or_simulate(config, alpha, ih)
            assert state is not None
            ax = axes[row_idx, col_idx]

            seq_num += 1

            # build triangulation and plot field directly to avoid Drawer autoscaling issues
            nodes = state.body.mesh.nodes
            tris = state.body.mesh.elements
            vals = np.asarray(state.temperature).ravel()

            triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)

            # contour lines and filled contour
            ax.tricontour(nodes[:, 0], nodes[:, 1], tris, vals, 15, colors="k", linewidths=0.2)
            n_layers = 100
            cf = ax.tricontourf(
                nodes[:, 0],
                nodes[:, 1],
                tris,
                vals,
                n_layers,
                cmap="plasma",
                vmin=field_min,
                vmax=field_max,
            )

            # mesh outline
            ax.triplot(triang, "k-", alpha=0.15, linewidth=0.3)

            inf_symbol = r"$\infty$"
            ax.set_title(
                string.ascii_lowercase[seq_num - 1]
                + ") "
                + rf"$\alpha$={alpha if alpha != np.inf else inf_symbol}, h=1/{ih}"
            )
            ax.set_aspect("equal", adjustable="box")
            # set axis limits to nodes extents
            x_min, x_max = float(nodes[:, 0].min()), float(nodes[:, 0].max())
            y_min, y_max = float(nodes[:, 1].min()), float(nodes[:, 1].max())
            dx, dy = x_max - x_min, y_max - y_min
            ax.set_xlim(x_min - 0.05 * dx, x_max + 0.05 * dx)
            ax.set_ylim(y_min - 0.05 * dy, y_max + 0.05 * dy)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")

            # release heavy state object before moving to next subplot
            del state
            gc.collect()

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=field_min, vmax=field_max))
    sm.set_array([])
    # leave space at the bottom for a horizontal colorbar
    fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25)
    fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        label="temperature",
        fraction=0.04,
        pad=0.08,
    )

    if config.save:
        fig.savefig(
            Path(config.outputs_path) / "tarzia_problem_grid.png",
            bbox_inches="tight",
            dpi=300,
        )
    if config.show:
        plt.show()
    plt.close(fig)


def draw_convergence_plots(config, sequences, ihs, alphas):
    rows = len(sequences)
    cols = len(sequences[0]) if sequences else 1

    # Check if last row has a single plot (needs centering)
    last_row_has_single = (
        len(sequences) > 0 and len(sequences[-1]) == 2 and sequences[-1][1] is None
    )

    # Create GridSpec with centering for single plot in last row
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(7 * cols, 4.5 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    axes = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            if r == rows - 1 and c == 0 and last_row_has_single:
                # Center last plot by using middle columns
                ax = fig.add_subplot(gs[r, :])
            else:
                ax = fig.add_subplot(gs[r, c])
            axes[r, c] = ax

    seq_num = 0
    for row_idx, row in enumerate(sequences):
        for col_idx, sequence in enumerate(row):
            if sequence is None:
                axes[row_idx, col_idx].axis("off")
                continue

            seq_num += 1
            ax = axes[row_idx, col_idx]
            print(f"Convergence sequence {seq_num}")

            reference_alpha, reference_ih = sequence[-1]
            reference_state = load_or_simulate(config, reference_alpha, reference_ih)
            assert reference_state is not None

            try:
                errors = []
                labels = []
                for alpha, ih in sequence[:-1]:
                    print(f"  loading {alpha=}, {ih=}")
                    state = load_or_simulate(config, alpha, ih)
                    assert state is not None
                    try:
                        errors.append(_relative_temperature_error(reference_state, state))
                        labels.append(_pair_label(alpha, ih))
                    finally:
                        del state
                        gc.collect()

                x = np.arange(len(sequence) - 1)
                y = np.maximum(np.asarray(errors, dtype=float), 1e-16)
                ax.semilogy(x, y, marker="o", linewidth=1.8, markersize=5, color="black")
                ax.set_xticks(x, labels, rotation=25, ha="right")
                ax.set_ylabel("$L_2$ norm between solutions")
                ax.grid(True, which="both", alpha=0.3)
                ax.set_title(
                    f"{string.ascii_lowercase[seq_num - 1]}) "
                    f"{_pair_label(*sequence[0])} → {_pair_label(*sequence[-1])}"
                )
                ax.set_xlabel(r"$(\alpha, h)$")
            finally:
                del reference_state
                gc.collect()

    fig.tight_layout()

    # LaTeX table output: errors w.r.t. reference finest mesh (alpha=inf)
    try:
        ref_ih = max(ihs)
        ref_inf_state = load_or_simulate(config, np.inf, ref_ih)
        assert ref_inf_state is not None

        alpha_comp = 1_000_000 if not config.test else alphas[0]
        errors_inf = []
        errors_alpha = []
        hs = []
        for ih in ihs:
            hs.append(1.0 / ih)
            s_inf = load_or_simulate(config, np.inf, ih)
            s_alpha = load_or_simulate(config, alpha_comp, ih)
            try:
                err_inf = (
                    _relative_temperature_error(ref_inf_state, s_inf)
                    if s_inf is not None
                    else float("nan")
                )
                # second column compares u_alpha^h to u_inf^{h_ref}
                err_alpha = (
                    _relative_temperature_error(ref_inf_state, s_alpha)
                    if s_alpha is not None
                    else float("nan")
                )
            finally:
                if s_inf is not None:
                    del s_inf
                if s_alpha is not None:
                    del s_alpha
                gc.collect()
            errors_inf.append(err_inf)
            errors_alpha.append(err_alpha)

        # compute rates (log-log) between consecutive h values
        def compute_rates(errs, hs):
            rates = ["--"]
            for i in range(1, len(errs)):
                e_prev, e_curr = errs[i - 1], errs[i]
                h_prev, h_curr = hs[i - 1], hs[i]
                if e_prev > 0 and e_curr > 0:
                    rate = np.log(e_prev / e_curr) / np.log(h_prev / h_curr)
                    rates.append(f"{rate:.2f}")
                else:
                    rates.append("--")
            return rates

        rates_inf = compute_rates(errors_inf, hs)
        rates_alpha = compute_rates(errors_alpha, hs)

        # print LaTeX table
        print("\\begin{table}[ht]")
        print("\\centering")
        print("\\begin{tabular}{|c|c|c|c|c|}")
        print("\\hline")
        header = (
            "$h$ & $\\|u_{\\infty}^h - u_{\\infty}^{h_{\\rm ref}}\\|_{L^2}$"
            " & Rate & $\\|u_{\\alpha}^h - u_{\\infty}^{h_{\\rm ref}}\\|_{L^2}$ ($\\alpha=10^6$) & Rate \\\\"
        )
        print(header)
        print("\\hline")
        for ih, err_i, r_i, err_a, r_a in zip(
            ihs, errors_inf, rates_inf, errors_alpha, rates_alpha
        ):
            h_str = f"$1/{ih}$"
            err_i_str = f"{err_i:.3e}" if not np.isnan(err_i) else "nan"
            err_a_str = f"{err_a:.3e}" if not np.isnan(err_a) else "nan"
            # each printed row must end with two backslashes for LaTeX linebreak
            # build row and append literal '\\' for LaTeX linebreak
            row = f"{h_str}  & {err_i_str} & {r_i}  & {err_a_str} & {r_a}  " + "\\\\"
            print(row)
        print("\\hline")
        print("\\end{tabular}")
        print(
            "\\caption{Relative $L^2$ errors and estimated convergence rates with respect to mesh size $h$ "
            "(reference solution: finest mesh at corresponding $\\alpha$).}"
        )
        print("\\label{tab:error_h}")
        print("\\end{table}")

    except Exception as e:
        print(f"Failed to produce LaTeX table: {e}")

    if config.save:
        fig.savefig(
            Path(config.outputs_path) / "tarzia_convergence_plots.png",
            bbox_inches="tight",
            dpi=300,
        )
    if config.show:
        plt.show()
    plt.close(fig)
