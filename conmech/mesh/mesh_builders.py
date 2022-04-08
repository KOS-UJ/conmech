from typing import Tuple

import numpy as np

from conmech.helpers import mph
from conmech.mesh import mesh_builders_legacy, mesh_builders_2d, mesh_builders_3d
from conmech.properties.mesh_properties import MeshProperties


def build_mesh(
        mesh_data: MeshProperties,
        create_in_subprocess=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if "cross" in mesh_data.mesh_type:
        return mesh_builders_legacy.get_cross_rectangle(mesh_data)

    if "meshzoo" in mesh_data.mesh_type:
        if "3d" in mesh_data.mesh_type:
            if "cube" in mesh_data.mesh_type:
                return mesh_builders_3d.get_meshzoo_cube(mesh_data)
            if "ball" in mesh_data.mesh_type:
                return mesh_builders_3d.get_meshzoo_ball(mesh_data)
        else:
            return mesh_builders_2d.get_meshzoo_rectangle(mesh_data)

    if "pygmsh" in mesh_data.mesh_type:
        if "3d" in mesh_data.mesh_type:
            if "polygon" in mesh_data.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_polygon(
                    mesh_data)
            if "twist" in mesh_data.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_twist(mesh_data)
        else:
            inner_function = lambda: mesh_builders_2d.get_pygmsh_elements_and_nodes(
                mesh_data
            )

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()

    raise NotImplementedError(f"Not implemented mesh type: {mesh_data.mesh_type}")
