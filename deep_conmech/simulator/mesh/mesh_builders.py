from ctypes import ArgumentError
from conmech.helpers import mph
from deep_conmech.simulator.mesh import mesh_builders_legacy, mesh_builders_2d, mesh_builders_3d


def build_mesh(
    mesh_type,
    mesh_density_x,
    mesh_density_y=None,
    scale_x=None,
    scale_y=None,
    is_adaptive=None,
    create_in_subprocess=False,
):
    if mesh_type == "cross":
        return mesh_builders_legacy.get_cross_rectangle(
            mesh_density_x, mesh_density_y, scale_x, scale_y
        )
    elif "meshzoo" in mesh_type:
        if "3d" in mesh_type:
            if "cube" in mesh_type:
                return mesh_builders_3d.get_meshzoo_cube(mesh_density_x)
            if "ball" in mesh_type:
                return mesh_builders_3d.get_meshzoo_ball(mesh_density_x)
        else:
            return mesh_builders_2d.get_meshzoo_rectangle(mesh_density_x, scale_x, scale_y)

    elif "dmsh" in mesh_type:
        return mesh_builders_2d.get_dmsh_rectangle(mesh_density_x, scale_x, scale_y)

    elif "pygmsh" in mesh_type:
        if "3d" in mesh_type:
            # initial_nodes, elements = get_twist(mesh_size)
            inner_function = lambda: mesh_builders_3d.get_pygmsh_extrude(mesh_density_x)
        else:
            inner_function = lambda: mesh_builders_2d.get_pygmsh_elements_and_nodes(
                mesh_type, mesh_density_x, scale_x, scale_y, is_adaptive
            )

        return mph.run_process(inner_function) if create_in_subprocess else inner_function()
        
    else:
        raise ArgumentError()

