from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

from conmech.helpers import cmh
from conmech.helpers.config import Config
from deep_conmech.scenarios import Scenario, TemperatureScenario
from deep_conmech.simulator.setting.setting_iterable import SettingIterable

# TODO: Move to config
dpi = 800
savefig_args = dict(transparent=False, facecolor="#191C20", pad_inches=0.0)  # "#24292E"


@dataclass
class ColorbarSettings:
    vmin: float
    vmax: float
    cmap: ListedColormap

    @property
    def mappable(self):
        norm = matplotlib.colors.Normalize(vmin=self.vmin, vmax=self.vmax)
        return plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)


def get_t_scale(scenario: Scenario, plot_setting_paths: List[str]):
    if isinstance(scenario, TemperatureScenario) is False:
        return None
    temperatures = np.array(
        [SettingIterable.load_pickle(path).t_old for path in plot_setting_paths]
    )
    return np.array([np.min(temperatures), np.max(temperatures)])


def get_t_data(t_scale: np.ndarray) -> ColorbarSettings:
    # magma plasma cool coolwarm
    lim_small = 0.2
    lim_big = 10

    if (t_scale[0] > -lim_small and t_scale[1] < lim_small):
        return ColorbarSettings(vmin=-lim_small, vmax=lim_small, cmap=plt.cm.cool)  # coolwarm
    return ColorbarSettings(vmin=-lim_big, vmax=lim_big, cmap=plt.cm.magma)


def plot_colorbar(fig, axs, cbar_settings):
    for ax in axs:
        position = ax.get_position()
        if (position.p0[0] > 0.1):
            position.p0[0] *= 0.8
        position.p1[0] *= 0.9
        ax.set_position(position)

    ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    set_ax(ax)
    cbar = fig.colorbar(mappable=cbar_settings.mappable, cax=ax)
    cbar.outline.set_edgecolor("w")
    cbar.outline.set_linewidth(0.2)


def set_ax(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor("w")
        spine.set_linewidth(0.2)
    ax.tick_params(color='w', labelcolor='w', width=0.3, labelsize=5)


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
        plot_setting_paths: List[str],
        time_skip: float,
        save_path: str,
        get_axs: Callable,
        plot_frame: Callable,
        fig,
        config: Config,
        t_scale: Optional[np.ndarray] = None
):
    # frac_skip = config.PRINT_SKIP
    # skip = int(frac_skip // scenario.time_step)
    frames_count = len(plot_setting_paths)  # // skip
    fps = int(1 / time_skip)
    animation_tqdm = cmh.get_tqdm(iterable=range(frames_count + 1), config=config,
                                  desc="Generating animation")

    def animate(step):
        current_time = step * time_skip
        animation_tqdm.update(1)
        fig.clf()
        axs = get_axs(fig)
        path = plot_setting_paths[step]
        setting = SettingIterable.load_pickle(path)
        plot_frame(axs=axs, fig=fig, setting=setting, current_time=current_time, t_scale=t_scale)
        return fig

    ani = animation.FuncAnimation(
        fig, animate, frames=frames_count
    )  # , interval=scenario.final_time)
    ani.save(save_path, writer=None, fps=fps, dpi=dpi, savefig_kwargs=savefig_args)
    # animation_tqdm.close()
    plt.close()


def get_frame_annotation(setting, current_time):
    return f"""time: {str(round(current_time, 1))}
nodes: {str(setting.nodes_count)}"""
