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
"""
Command line entry shared by the examples.
"""

import argparse
import shutil
from pathlib import Path
from typing import Callable, Optional

from conmech.helpers.config import Config


def default_cleanup(outputs_path: str) -> int:
    """
    Remove an example's output directory. Returns the number of files removed.
    """
    directory = Path(outputs_path)
    if not directory.exists():
        print(f"nothing to clean in {directory}")
        return 0
    removed = sum(1 for path in directory.rglob("*") if path.is_file())
    shutil.rmtree(directory)
    print(f"removed {removed} files from {directory}")
    return removed


def build_parser(default_outputs_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a conmech example.")
    parser.add_argument(
        "--outputs-path", default=default_outputs_path, help="where results and figures go"
    )
    parser.add_argument("--show", action="store_true", help="display figures instead of saving")
    parser.add_argument("--no-save", action="store_true", help="do not write figures to disk")
    parser.add_argument(
        "--force", action="store_true", help="recompute even when a stored result is current"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove the output directory and exit, running no simulation",
    )
    parser.add_argument(
        "--test", action="store_true", help="the small configuration used by the test suite"
    )
    return parser


def run_example(
    main: Callable[[Config], None],
    setup_module,
    cleanup: Optional[Callable[[str], int]] = None,
    argv: Optional[list] = None,
) -> None:
    default_outputs_path = getattr(setup_module, "OUTPUTS_PATH", "./output")
    arguments = build_parser(default_outputs_path).parse_args(argv)

    if arguments.cleanup:
        (cleanup or default_cleanup)(arguments.outputs_path)
        return

    config = Config(
        outputs_path=arguments.outputs_path,
        show=arguments.show,
        save=not arguments.show and not arguments.no_save,
        force=arguments.force,
        test=arguments.test,
    ).init()
    main(config)
