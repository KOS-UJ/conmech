from ctypes import ArgumentError

from conmech.dataclass.mesh_data import MeshData
from conmech.helpers import mph
from deep_conmech.simulator.mesh import mesh_builders_legacy, mesh_builders_2d, mesh_builders_3d


def build_mesh(
        mesh_data: MeshData,
        create_in_subprocess=False,
):
    if "cross" in mesh_data.mesh_type:
        return mesh_builders_legacy.get_cross_rectangle(
            mesh_data.mesh_density_x, mesh_data.mesh_density_y, mesh_data.scale_x, mesh_data.scale_y
        )
    elif "meshzoo" in mesh_data.mesh_type:
        if "3d" in mesh_data.mesh_type:
            if "cube" in mesh_data.mesh_type:
                return mesh_builders_3d.get_meshzoo_cube(mesh_data.mesh_density_x)
            if "ball" in mesh_data.mesh_type:
                return mesh_builders_3d.get_meshzoo_ball(mesh_data.mesh_density_x)
        else:
            return mesh_builders_2d.get_meshzoo_rectangle(mesh_data)

    elif "dmsh" in mesh_data.mesh_type:
        return mesh_builders_2d.get_dmsh_rectangle(mesh_data)

    elif "pygmsh" in mesh_data.mesh_type:
        if "3d" in mesh_data.mesh_type:
            if "polygon" in mesh_data.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_polygon(
                    mesh_data.mesh_density_x)
            if "twist" in mesh_data.mesh_type:
                inner_function = lambda: mesh_builders_3d.get_pygmsh_twist(mesh_data.mesh_density_x)
        else:
            inner_function = lambda: mesh_builders_2d.get_pygmsh_elements_and_nodes(
                mesh_data
            )

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()

    else:
        raise ArgumentError()
