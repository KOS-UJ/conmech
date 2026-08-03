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
Reuse of simulation results across runs, keyed on the example's `setup` module.

An example keeps every parameter in its `setup` module, so the content of that
one file decides what a simulation means. The cache stores its digest next to
each result and recomputes when the two disagree. Parameters that a study sweeps
over, such as the mesh size or a penalty coefficient, go into the file name
instead, so one setup can own many results.
"""

import hashlib
import inspect
import json
import pickle
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional


class SetupChanged(RuntimeError):
    """A stored result was produced by a different `setup` module."""


def setup_digest(setup_module) -> str:
    """
    Digest of the source file of `setup_module`.
    """
    path = Path(inspect.getsourcefile(setup_module))
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _name_from(prefix: str, parameters: Dict) -> str:
    parts = [prefix] if prefix else []
    for key in sorted(parameters):
        value = parameters[key]
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        parts.append(f"{key}_{value}")
    return "_".join(parts) or "result"


class SimulationCache:

    def __init__(self, setup_module, outputs_path: str, prefix: str = ""):
        self.setup_module = setup_module
        self.outputs_path = Path(outputs_path)
        self.prefix = prefix
        self.digest = setup_digest(setup_module)

    def path(self, **parameters) -> Path:
        """Path of the result for these parameters."""
        return self.outputs_path / _name_from(self.prefix, parameters)

    def meta_path(self, **parameters) -> Path:
        path = self.path(**parameters)
        return path.with_name(path.name + ".meta.json")

    def is_current(self, **parameters) -> bool:
        """True when a stored result exists and matches the current setup."""
        path, meta = self.path(**parameters), self.meta_path(**parameters)
        if not path.exists() or not meta.exists():
            return False
        stored = json.loads(meta.read_text(encoding="utf-8"))
        return stored.get("setup_digest") == self.digest

    def load(self, **parameters):
        """Load a stored result, refusing one that a different setup produced."""
        if not self.is_current(**parameters):
            raise SetupChanged(
                f"{self.path(**parameters)} is missing or was produced by a different "
                f"setup than {inspect.getsourcefile(self.setup_module)}"
            )
        with open(self.path(**parameters), "rb") as handle:
            return pickle.load(handle)

    def save(self, result, extra_meta: Optional[Dict] = None, **parameters) -> Path:
        path = self.path(**parameters)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(result, handle)
        meta = {
            "setup_digest": self.digest,
            "setup_file": str(Path(inspect.getsourcefile(self.setup_module)).name),
            "parameters": {key: str(value) for key, value in parameters.items()},
        }
        meta.update(extra_meta or {})
        self.meta_path(**parameters).write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        return path

    def load_or_compute(self, compute: Callable, force: bool = False, **parameters):
        """
        Return the stored result, or compute and store it.

        `compute` is called with the parameters as keyword arguments and
        must return a picklable object.
        """
        if not force and self.is_current(**parameters):
            return self.load(**parameters)
        result = compute(**parameters)
        self.save(result, **parameters)
        return result

    def cleanup(self) -> int:
        """
        Remove this example's output directory. Returns the number of files
        removed, so a caller can report what it freed.
        """
        if not self.outputs_path.exists():
            return 0
        removed = sum(1 for path in self.outputs_path.rglob("*") if path.is_file())
        shutil.rmtree(self.outputs_path)
        return removed
