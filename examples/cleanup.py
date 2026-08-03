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
Remove the results of every example.

Each example package declares where it writes through `OUTPUTS_PATH` in its
`setup` module; this collects them all, plus the shared default, and removes
what is there. A single example cleans up after itself with
`python -m examples.<name>.run --cleanup`.

    python -m examples.cleanup --dry-run    # list what would go
    python -m examples.cleanup              # ask, then remove
    python -m examples.cleanup --yes        # remove without asking
"""

import argparse
import importlib
import pkgutil
import shutil
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_OUTPUTS = ["./output"]


def discover_outputs_paths() -> Dict[str, str]:
    import examples

    found = {}
    for module in pkgutil.iter_modules(examples.__path__):
        if not module.ispkg or module.name == "common":
            continue
        try:
            setup = importlib.import_module(f"examples.{module.name}.setup")
        except ImportError as error:  # an example whose dependencies are absent
            print(f"  skipping {module.name}: {error}", file=sys.stderr)
            continue
        outputs_path = getattr(setup, "OUTPUTS_PATH", None)
        if outputs_path:
            found[module.name] = outputs_path
    return found


def collect_targets(extra: List[str]) -> Dict[str, Path]:
    candidates = dict(discover_outputs_paths())
    for path in list(DEFAULT_OUTPUTS) + list(extra):
        candidates.setdefault(f"(default) {path}", path)

    resolved = {}
    for name, path in candidates.items():
        directory = Path(path).resolve()
        if directory.is_dir():
            resolved.setdefault(directory, name)

    targets = {}
    for directory, name in resolved.items():
        if any(other != directory and other in directory.parents for other in resolved):
            continue
        targets[name] = directory
    return targets


def count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Remove the results of every example.")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be removed and stop"
    )
    parser.add_argument("--yes", action="store_true", help="remove without asking")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="an extra directory to clean, repeatable",
    )
    arguments = parser.parse_args(argv)

    targets = collect_targets(arguments.path)
    if not targets:
        print("nothing to clean")
        return 0

    total = 0
    for name, directory in sorted(targets.items()):
        files = count_files(directory)
        total += files
        print(f"  {directory}  ({files} files)  <- {name}")
    print(f"{len(targets)} directories, {total} files")

    if arguments.dry_run:
        return 0

    if not arguments.yes:
        if not sys.stdin.isatty():
            print("refusing to remove without --yes when not attached to a terminal")
            return 1
        if input("remove these directories? [y/N] ").strip().lower() not in ("y", "yes"):
            print("nothing removed")
            return 0

    for directory in targets.values():
        shutil.rmtree(directory)
    print(f"removed {total} files from {len(targets)} directories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
