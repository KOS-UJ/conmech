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
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from examples.BartmanSzwarc_Ochal_Tarzia_2026.run import load_or_simulate
from examples.BartmanSzwarc_Ochal_Tarzia_2026.setup import (
    ExampleSpec,
    alpha_tag,
    analytic_gap_l2,
    analytic_gap_v,
    mesh_size,
)
from examples.common import error_norms as err
from examples.common.table_exporter import export_table, format_error, format_rate


def _alpha_math(alpha: float) -> str:
    if alpha_tag(alpha) == "inf":
        return r"\infty"
    exponent = int(round(np.log10(alpha)))
    if np.isclose(alpha, 10.0**exponent):
        return f"10^{{{exponent}}}"
    return f"{alpha:g}"


def _errors_of(state, u_exact, grad_exact) -> Dict:
    return err.errors_vs_exact(
        state.body.mesh.nodes, state.body.mesh.elements, state.temperature, u_exact, grad_exact
    )


def table_vs_exact(config, spec: ExampleSpec, alpha, ihs: Sequence[int], table_id: str) -> Dict:
    """
    Errors of `u^h` against the closed-form solution, with the rates in `h`.

    `L^2` should approach 2 and the `H^1` seminorm 1 for a smooth solution.
    """
    u_exact, grad_exact = spec.exact_for(alpha)
    if u_exact is None:
        raise ValueError(f"{spec.name} has no closed-form solution for alpha={alpha}")

    hs, l2s, h1s, vs, linfs = [], [], [], [], []
    for ih in ihs:
        state = load_or_simulate(config, spec, alpha, ih)
        try:
            measured = _errors_of(state, u_exact, grad_exact)
        finally:
            del state
            gc.collect()
        hs.append(mesh_size(ih))
        l2s.append(measured["L2"])
        h1s.append(measured["H1_semi"])
        vs.append(measured["V"])
        linfs.append(measured["Linf_nodal"])

    rates_l2, rates_h1, rates_v = err.rates_h(l2s, hs), err.rates_h(h1s, hs), err.rates_h(vs, hs)
    rows = [
        [
            f"$1/{ih}$",
            format_error(l2s[i]),
            format_rate(rates_l2[i]),
            format_error(h1s[i]),
            format_rate(rates_h1[i]),
            format_error(vs[i]),
            format_rate(rates_v[i]),
            format_error(linfs[i]),
        ]
        for i, ih in enumerate(ihs)
    ]
    export_table(
        table_id,
        [
            "$h$",
            r"$\|u^h-u\|_{L^2}$",
            "rate",
            r"$\|\nabla(u^h-u)\|_{L^2}$",
            "rate",
            r"$\|u^h-u\|_{V}$",
            "rate",
            r"$\|u^h-u\|_{\infty,\rm nodal}$",
        ],
        rows,
        caption=(
            f"Example {spec.name}: errors of the discrete solution against the closed-form "
            f"solution for $\\alpha={_alpha_math(alpha)}$, and the estimated rates in $h$."
        ),
        label=f"tab:{spec.name}_alpha_{alpha_tag(alpha)}",
        outputs_path=config.outputs_path,
        table_name=f"table_{table_id}_{spec.name}",
    )
    return {"hs": hs, "L2": l2s, "H1_semi": h1s, "V": vs, "Linf_nodal": linfs}


def table_alpha_gap(config, spec: ExampleSpec, ih: int, alphas: Sequence[float]) -> Dict:
    state_inf = load_or_simulate(config, spec, np.inf, ih)
    l2s, vs, analytic, relative = [], [], [], []
    try:
        for alpha in alphas:
            state = load_or_simulate(config, spec, alpha, ih)
            try:
                measured = err.errors_between_states(
                    state, state_inf, context=f"{spec.name} alpha={alpha} ih={ih}"
                )
            finally:
                del state
                gc.collect()
            l2s.append(measured["L2"])
            vs.append(measured["V"])
            expected = analytic_gap_l2(alpha)
            analytic.append(expected)
            relative.append(abs(measured["L2"] - expected) / expected if expected else np.nan)
    finally:
        del state_inf
        gc.collect()

    rates = err.rates_alpha(l2s, alphas)
    rows = [
        [
            f"${_alpha_math(alphas[i])}$",
            format_error(l2s[i]),
            format_error(vs[i]),
            format_rate(rates[i]),
            format_error(analytic[i]),
            f"{100 * relative[i]:.2f}\\%",
        ]
        for i in range(len(alphas))
    ]
    export_table(
        "C",
        [
            r"$\alpha$",
            r"$\|u^h_\alpha-u^h_\infty\|_{L^2}$",
            r"$\|u^h_\alpha-u^h_\infty\|_{V}$",
            "rate",
            r"$\frac{7}{1+\alpha}\sqrt{2/3}$",
            "rel. dev.",
        ],
        rows,
        caption=(
            f"Example {spec.name}, $h=1/{ih}$: distance between the penalised and the limit "
            "discrete solutions, its order in $\\alpha$, and the analytic value."
        ),
        label=f"tab:{spec.name}_alpha_order",
        outputs_path=config.outputs_path,
        table_name=f"table_C_{spec.name}",
    )
    return {"alphas": list(alphas), "L2": l2s, "V": vs, "analytic": analytic, "rel": relative}


def table_double_limit(
    config, spec: ExampleSpec, ihs: Sequence[int], alphas: Sequence[float]
) -> np.ndarray:
    u_exact, grad_exact = spec.exact_for(np.inf)
    matrix = np.full((len(ihs), len(alphas)), np.nan)
    for row, ih in enumerate(ihs):
        for col, alpha in enumerate(alphas):
            state = load_or_simulate(config, spec, alpha, ih)
            try:
                matrix[row, col] = _errors_of(state, u_exact, grad_exact)["V"]
            finally:
                del state
                gc.collect()

    rows = [
        [f"$1/{ih}$"] + [format_error(matrix[row, col]) for col in range(len(alphas))]
        for row, ih in enumerate(ihs)
    ]
    export_table(
        "D",
        ["$h$"] + [f"${_alpha_math(a)}$" for a in alphas],
        rows,
        caption=(
            f"Example {spec.name}: $\\|u^h_\\alpha-u_\\infty\\|_{{V}}$ against the mesh size and "
            "the penalty parameter."
        ),
        label=f"tab:{spec.name}_double_limit",
        outputs_path=config.outputs_path,
        table_name=f"table_D_{spec.name}",
    )
    return matrix


ALPHA_PATHS = (
    (r"$\alpha = 1/h$", lambda ih: float(ih)),
    (r"$\alpha = 1/h^2$", lambda ih: float(ih) ** 2),
    (r"$\alpha = 100$ (fixed)", lambda ih: 100.0),
)


def figure_alpha_paths(config, spec: ExampleSpec, ihs: Sequence[int]) -> Dict:
    u_exact, grad_exact = spec.exact_for(np.inf)
    hs = np.array([mesh_size(ih) for ih in ihs])

    curves: Dict[str, List[float]] = {}
    for label, alpha_of in ALPHA_PATHS:
        values = []
        for ih in ihs:
            state = load_or_simulate(config, spec, alpha_of(ih), ih)
            try:
                values.append(_errors_of(state, u_exact, grad_exact)["V"])
            finally:
                del state
                gc.collect()
        curves[label] = values

    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    for (label, _), marker in zip(ALPHA_PATHS, ["o", "s", "^"]):
        axis.loglog(hs, curves[label], marker=marker, linewidth=1.8, label=label)
    reference = np.asarray(curves[ALPHA_PATHS[1][0]], dtype=float)
    axis.loglog(hs, reference[0] * (hs / hs[0]), "k--", linewidth=1.0, label=r"$O(h)$")
    axis.loglog(hs, reference[0] * (hs / hs[0]) ** 2, "k:", linewidth=1.0, label=r"$O(h^2)$")
    axis.set_xlabel("$h$")
    axis.set_ylabel(r"$\|u^h_\alpha-u_\infty\|_{V}$")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    axis.set_title(f"Example {spec.name}: paths in the $(h,\\alpha)$ plane")
    figure.tight_layout()
    _finish_figure(config, figure, f"figure_E_{spec.name}.png")

    _report_stagnation(curves[ALPHA_PATHS[2][0]], hs, alpha_fixed=100.0)
    return {"hs": hs.tolist(), "curves": curves}


def _report_stagnation(values: Sequence[float], hs: Sequence[float], alpha_fixed: float) -> None:
    """
    Only the tail can show it: on the coarsest meshes the discretisation error
    dominates. What must hold is that the last rate collapses towards zero and
    that the curve never drops below the floor.
    """
    floor = analytic_gap_v(alpha_fixed)
    last_rate = err.rate_h(values[-2], values[-1], hs[-2], hs[-1]) if len(values) > 1 else np.nan
    print(
        f"% figure E: fixed alpha={alpha_fixed:g} ends at {values[-1]:.3e} "
        f"(last rate in h: {last_rate:.2f}); analytic floor {floor:.3e}"
    )
    if values[-1] < 0.8 * floor:
        print(
            f"% WARNING: the fixed-alpha path fell below its floor {floor:.3e}; either the penalty "
            f"is not scaled by alpha, or the limit solution is wrong."
        )
    elif np.isfinite(last_rate) and last_rate > 0.7:
        print(
            f"% WARNING: the fixed-alpha path still falls at rate {last_rate:.2f} and has not "
            "stagnated. Either the mesh range is too coarse for the penalty error to dominate, "
            "or alpha is not held fixed."
        )


def figure_gamma3_trace(config, spec: ExampleSpec, ih: int, alphas: Sequence[float]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.0))
    for alpha in alphas:
        state = load_or_simulate(config, spec, alpha, ih)
        try:
            nodes = np.asarray(state.body.mesh.nodes, dtype=float)
            values = np.asarray(state.temperature, dtype=float).ravel()
            on_top = np.nonzero(np.isclose(nodes[:, 1], 1.0, atol=1e-9))[0]
            order = on_top[np.argsort(nodes[on_top, 0])]
            axis.plot(
                nodes[order, 0],
                values[order],
                marker=".",
                linewidth=1.4,
                label=rf"$\alpha={_alpha_math(alpha)}$",
            )
        finally:
            del state
            gc.collect()
    axis.axhline(spec.b, color="k", linestyle="--", linewidth=1.2, label=f"$u=b={spec.b:g}$")
    axis.set_xlabel("$x$")
    axis.set_ylabel(r"$u^h_\alpha(x,1)$")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    axis.set_title(rf"Example {spec.name}: trace on $\Gamma_3$, $h=1/{ih}$")
    figure.tight_layout()
    _finish_figure(config, figure, f"figure_F_{spec.name}.png")


def _finish_figure(config, figure, name: str) -> None:
    if config.save and config.outputs_path:
        from pathlib import Path

        Path(config.outputs_path).mkdir(parents=True, exist_ok=True)
        figure.savefig(Path(config.outputs_path) / name, bbox_inches="tight", dpi=300)
    if config.show:
        plt.show()
    plt.close(figure)


def draw_temperature_grid(config, spec: ExampleSpec, to_plot) -> None:
    rows, cols = len(to_plot), len(to_plot[0])
    figure, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)

    field_min, field_max = float("inf"), float("-inf")
    for row in to_plot:
        for alpha, ih in row:
            state = load_or_simulate(config, spec, alpha, ih)
            try:
                temperature = np.asarray(state.temperature, dtype=float).ravel()
                field_min = min(field_min, float(temperature.min()))
                field_max = max(field_max, float(temperature.max()))
            finally:
                del state
                gc.collect()

    seq_num = 0
    for row_idx, row in enumerate(to_plot):
        for col_idx, (alpha, ih) in enumerate(row):
            state = load_or_simulate(config, spec, alpha, ih)
            try:
                axis = axes[row_idx, col_idx]
                seq_num += 1
                nodes = np.asarray(state.body.mesh.nodes, dtype=float)
                tris = np.asarray(state.body.mesh.elements, dtype=np.int64)
                values = np.asarray(state.temperature, dtype=float).ravel()
                triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris)
                axis.tricontour(triangulation, values, 15, colors="k", linewidths=0.2)
                axis.tricontourf(
                    triangulation, values, 100, cmap="plasma", vmin=field_min, vmax=field_max
                )
                axis.triplot(triangulation, "k-", alpha=0.15, linewidth=0.3)
                axis.set_title(
                    string.ascii_lowercase[seq_num - 1]
                    + ") "
                    + rf"$\alpha$={_alpha_math(alpha)}, h=1/{ih}"
                )
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("$x$")
                axis.set_ylabel("$y$")
            finally:
                del state
                gc.collect()

    scalar_map = plt.cm.ScalarMappable(
        cmap="plasma", norm=plt.Normalize(vmin=field_min, vmax=field_max)
    )
    scalar_map.set_array([])
    figure.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25)
    figure.colorbar(
        scalar_map,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        label="temperature",
        fraction=0.04,
        pad=0.08,
    )
    _finish_figure(config, figure, f"temperature_grid_{spec.name}.png")


def table_vs_reference(
    config,
    spec: ExampleSpec,
    alpha,
    ihs: Sequence[int],
    ih_ref: int,
    table_id: str = "H",
) -> Dict:
    reference = load_or_simulate(config, spec, alpha, ih_ref)
    hs, l2s, h1s, vs = [], [], [], []
    try:
        for ih in ihs:
            state = load_or_simulate(config, spec, alpha, ih)
            try:
                measured = err.errors_between_states(
                    state, reference, context=f"{spec.name} alpha={alpha} ih={ih} vs {ih_ref}"
                )
            finally:
                del state
                gc.collect()
            hs.append(mesh_size(ih))
            l2s.append(measured["L2"])
            h1s.append(measured["H1_semi"])
            vs.append(measured["V"])
    finally:
        del reference
        gc.collect()

    rates_l2, rates_h1, rates_v = err.rates_h(l2s, hs), err.rates_h(h1s, hs), err.rates_h(vs, hs)
    rows = [
        [
            f"$1/{ih}$",
            format_error(l2s[i]),
            format_rate(rates_l2[i]),
            format_error(h1s[i]),
            format_rate(rates_h1[i]),
            format_error(vs[i]),
            format_rate(rates_v[i]),
        ]
        for i, ih in enumerate(ihs)
    ]
    export_table(
        table_id,
        [
            "$h$",
            r"$\|u^h-u^{h_{\rm ref}}\|_{L^2}$",
            "rate",
            r"$\|\nabla(u^h-u^{h_{\rm ref}})\|_{L^2}$",
            "rate",
            r"$\|u^h-u^{h_{\rm ref}}\|_{V}$",
            "rate",
        ],
        rows,
        caption=(
            f"Example {spec.name}, $\\alpha={_alpha_math(alpha)}$: errors against the reference "
            f"solution on $h_{{\\rm ref}}=1/{ih_ref}$, and the estimated rates in $h$."
        ),
        label=f"tab:{spec.name}_ref_{alpha_tag(alpha)}",
        outputs_path=config.outputs_path,
        table_name=f"table_{table_id}_{spec.name}_alpha_{alpha_tag(alpha)}",
    )
    return {"hs": hs, "L2": l2s, "H1_semi": h1s, "V": vs}
