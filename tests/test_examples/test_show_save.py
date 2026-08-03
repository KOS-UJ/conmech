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
"""`show` and `save` are independent, and asking for both writes a real figure."""

import matplotlib
import pytest

matplotlib.use("Agg")

from conmech.helpers.config import Config
from conmech.plotting.drawer import Drawer


@pytest.fixture(name="state", scope="module")
def _state():
    from examples.example_static.run import simulate

    return simulate(2)


def draw(state, tmp_path, tag, show, save):
    outputs_path = tmp_path / tag
    config = Config(
        outputs_path=str(outputs_path), show=show, save=save, output_dir="figures"
    ).init()
    Drawer(state=state, config=config).draw(show=show, save=save)
    written = sorted(p for p in outputs_path.rglob("*") if p.is_file())
    return outputs_path, written


def test_show_only_writes_nothing_and_creates_no_directory(state, tmp_path):
    outputs_path, written = draw(state, tmp_path, "show_only", show=True, save=False)
    assert not outputs_path.exists(), "displaying results must not leave a directory behind"
    assert not written


def test_neither_show_nor_save_creates_no_directory(state, tmp_path):
    outputs_path, written = draw(state, tmp_path, "neither", show=False, save=False)
    assert not outputs_path.exists()
    assert not written


def test_save_only_writes_one_figure(state, tmp_path):
    _, written = draw(state, tmp_path, "save_only", show=False, save=True)
    assert len(written) == 1
    assert written[0].stat().st_size > 10_000


def test_show_and_save_writes_the_same_figure_as_save_alone(state, tmp_path):
    _, saved = draw(state, tmp_path, "save_only", show=False, save=True)
    _, both = draw(state, tmp_path, "show_save", show=True, save=True)

    assert len(both) == 1
    assert both[0].stat().st_size > 10_000
    assert both[0].read_bytes() == saved[0].read_bytes()


def test_config_allows_show_and_save_together():
    Config(show=True, save=True, outputs_path="/tmp/conmech-unused").init()
