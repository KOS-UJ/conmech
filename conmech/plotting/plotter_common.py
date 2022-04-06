from dataclasses import dataclass
from typing import Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.scenarios.scenarios import Scenario, TemperatureScenario
from matplotlib import animation
from matplotlib.colors import ListedColormap

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


def get_t_scale(scenario: Scenario, index_skip:int, plot_settings_count:int, all_settings_path: str):
    if isinstance(scenario, TemperatureScenario) is False:
        return None
    #TODO: Refactor (repetition from plot_animation)
    temperatures_list = []
    all_indices = pkh.get_all_indices_pickle(all_settings_path)
    settings_file = pkh.open_file_settings_read_pickle(all_settings_path)
    with settings_file:
        for step in range(plot_settings_count):
            setting = pkh.load_index_pickle(index=step * index_skip, all_indices=all_indices, settings_file=settings_file)
            temperatures_list.append(setting.t_old)
    temperatures = np.array(temperatures_list)
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
        get_axs: Callable,
        plot_frame: Callable,
        fig,
        save_path: str,
        config: Config,
        time_skip: float,
        index_skip: int,
        plot_settings_count: int,
        all_settings_path: str,
        t_scale: Optional[np.ndarray] = None,
):
    fps = int(1 / time_skip)
    animation_tqdm = cmh.get_tqdm(iterable=range(plot_settings_count + 1), config=config,
                                  desc="Generating animation")

    all_indices = pkh.get_all_indices_pickle(all_settings_path)
    settings_file = pkh.open_file_settings_read_pickle(all_settings_path)
    with settings_file:
        def animate(step):
            animation_tqdm.update(1)
            fig.clf()
            axs = get_axs(fig)
            setting = pkh.load_index_pickle(index=step * index_skip,
                                                        all_indices=all_indices,
                                                        settings_file=settings_file)
            plot_frame(axs=axs, fig=fig, setting=setting, current_time=step * time_skip,
                       t_scale=t_scale)
            return fig

        ani = animation.FuncAnimation(
            fig, animate, frames=plot_settings_count
        )  # , interval=scenario.final_time)

        ani.save(save_path, writer=None, fps=fps, dpi=dpi, savefig_kwargs=savefig_args)
        # animation_tqdm.close()
    plt.close()


def get_frame_annotation(setting, current_time):
    return f"""time: {str(round(current_time, 1))}
nodes: {str(setting.nodes_count)}"""
