import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import deep_conmech.common.config as config
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import cmh
from deep_conmech.scenarios import Scenario, TemperatureScenario
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_iterable import SettingIterable
from matplotlib import animation, cm, collections
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import LinearLocator

dpi = 800
savefig_args = dict(transparent=False, facecolor="#24292E", pad_inches=0.0)



@dataclass
class ColorbarSettings:
    vmin:float
    vmax:float
    cmap: ListedColormap

def get_t_scale(scenario: Scenario, all_setting_paths: List[str]):
    if isinstance(scenario, TemperatureScenario) is False:
        return None
    temperatures = np.array(
        [SettingIterable.load_pickle(path).t_old for path in all_setting_paths]
    )
    return np.array([np.min(temperatures), np.max(temperatures)])

def get_t_data(t_scale: np.ndarray) -> ColorbarSettings:
     # magma plasma cool coolwarm
    lim_small = 0.2
    lim_big = 10

    if(t_scale[0] > -lim_small and t_scale[1] < lim_small):
        return ColorbarSettings(vmin=-lim_small, vmax=lim_small, cmap=plt.cm.cool)
    return ColorbarSettings(vmin=-lim_big, vmax=lim_big, cmap=plt.cm.magma)

def plot_colorbar(fig, cbar_settings):
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    norm = matplotlib.colors.Normalize(vmin=cbar_settings.vmin, vmax=cbar_settings.vmax)
    values = plt.cm.ScalarMappable(norm=norm, cmap=cbar_settings.cmap)
    cbar = fig.colorbar(values, cax=cbar_ax)
    cbar.ax.tick_params(colors="w")

########

def prepare_for_arrows(starts, vectors):
    nodes_count = len(starts)
    scale = 1.0
    scaled_vectors = vectors * scale

    max_arrow_count = 64
    arrow_skip = 1
    if nodes_count > max_arrow_count:
        arrow_skip = int(nodes_count / max_arrow_count)

    mask_nonzero = scaled_vectors.any(axis=1)
    mask = [(i % arrow_skip == 0 and b) for i, b in enumerate(mask_nonzero)]

    return starts[mask].T, scaled_vectors[mask].T


def plt_save(path, extension):
    plt.savefig(
        path, **savefig_args, format=extension, dpi=dpi
    )  # , bbox_inches="tight"
    plt.close()


def plot_animation(
    all_setting_paths: List[str],
    time_skip: float,
    path: str,
    get_axs: Callable,
    plot_frame: Callable,
    fig,
    t_scale: Optional[np.ndarray] = None
):
    # frac_skip = config.PRINT_SKIP
    # skip = int(frac_skip // scenario.time_step)
    frames_count = len(all_setting_paths)  # // skip
    fps = int(1 / time_skip)
    animation_tqdm = cmh.get_tqdm(range(frames_count+1), desc="Generating animation")

    def animate(step):
        current_time = step * time_skip
        animation_tqdm.update(1)
        fig.clf()
        axs = get_axs(fig)
        path = all_setting_paths[step]
        setting = SettingIterable.load_pickle(path)
        plot_frame(axs=axs, fig=fig, setting=setting, current_time=current_time, t_scale=t_scale)
        return fig

    ani = animation.FuncAnimation(
        fig, animate, frames=frames_count
    )  # , interval=scenario.final_time)
    ani.save(path, writer=None, fps=fps, dpi=dpi, savefig_kwargs=savefig_args)
    #animation_tqdm.close()
    plt.close()
