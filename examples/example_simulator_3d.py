# %%
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import nph
from matplotlib.gridspec import GridSpec

from deep_conmech.common import config
from deep_conmech.common.plotter import plotter_3d
from deep_conmech.graph.helpers import thh
from deep_conmech import scenarios
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.matrices.matrices_3d import *
from deep_conmech.simulator.mesh.mesh_builders_3d import *
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles

catalog = f"output/3D - {thh.CURRENT_TIME}"



def main():
    mesh_density_x = 3
    # meshzoo_cube_3d
    # meshzoo_ball_3d
    # pygmsh_3d
    setting = SettingObstacles(
        mesh_type="meshzoo_cube_3d", mesh_density_x=mesh_density_x
    )
    setting.set_obstacles(scenarios.o_3d)

    all_images_paths = []
    extension = "png"  # pdf
    thh.create_folders(catalog)



    for i in range(1, config.EPISODE_STEPS + 1):
        current_time = i * setting.time_step
        print(f"time: {current_time}")

        forces = setting.get_forces_by_function(scenarios.f_rotate_3d, current_time)
        normalized_forces = setting.normalize_rotate(forces)
        setting.prepare(normalized_forces)

        normalized_a = Calculator.solve_normalized(setting)
        a = setting.denormalize_rotate(normalized_a)

        if i % 10 == 0:
            plotter_3d.plot_frame(
                setting=setting,
                normalized_data=[
                    normalized_forces * 20,
                    setting.normalized_u_old,
                    setting.normalized_v_old,
                    normalized_a,
                ],
                path=f"{catalog}/{int(thh.get_timestamp() * 100)}.{extension}",
                extension=extension,
                all_images_paths=all_images_paths,
            )

        setting.set_v_old(setting.v_old + setting.time_step * a)
        setting.set_u_old(setting.u_old + setting.time_step * setting.v_old)

    path = f"{catalog}/ANIMATION.gif"

    images = []
    for image_path in all_images_paths:
        images.append(imageio.imread(image_path))

    duration = 0.1
    args = {"duration": duration}
    imageio.mimsave(path, images, **args)

    for image_path in all_images_paths:
        os.remove(image_path)


if __name__ == "__main__":
    main()


# %%
