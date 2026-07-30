# CONMECH @ Jagiellonian University in Kraków
#
# Copyright (C) 2026  Piotr Bartman-Szwarc <piotr.bartman@uj.edu.pl>
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
"""Convergence tables as LaTeX on stdout and CSV on disk."""

import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ERROR_FMT = "%.3e"
RATE_FMT = "%.2f"


def format_error(value) -> str:
    """Format an error as `%.3e`; `--` for `None`, `nan` and `inf`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return ERROR_FMT % value


def format_rate(value) -> str:
    """Format a convergence rate as `%.2f`; `--` for `None`, `nan` and `inf`."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "--"
    return RATE_FMT % value


def export_table(
    table_id: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    outputs_path: str,
    table_name: Optional[str] = None,
) -> str:
    """
    Write a LaTeX code table to .tex file and raw data to .csv
    Returns `path/to/<table_name>` (WITHOUT extension!).
    """
    spec = "|".join(["c"] * len(headers))

    tex_lines = [
        f"% ---- Table {table_id} ----",
        "\\begin{table}[ht]",
        "\\centering",
        f"\\begin{{tabular}}{{|{spec}|}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]

    for row in rows:
        tex_lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])

    base_dir = Path(outputs_path) if outputs_path else Path(".")
    base_dir.mkdir(parents=True, exist_ok=True)

    table_name = table_name or f"table_{table_id}"

    tex_path = base_dir / (table_name + ".tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(tex_lines) + "\n")

    csv_path = base_dir / (table_name + ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)

    return str(base_dir / table_name)
