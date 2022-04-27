from dataclasses import dataclass
from io import BufferedReader
from typing import Callable, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.scenarios.scenarios import Scenario, TemperatureScenario

# TODO: #65 Move to config
DPI = 800
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


def get_t_scale(
    scenario: Scenario,
    index_skip: int,
    plot_scenes_count: int,
    all_scenes_path: str,
):
    if isinstance(scenario, TemperatureScenario) is False:
        return None
    # TODO: #65 Refactor (repetition from plot_animation)
    temperatures_list = []
    all_indices = pkh.get_all_indices(all_scenes_path)
    scenes_file = pkh.open_file_read(all_scenes_path)
    with scenes_file:
        for step in range(plot_scenes_count):
            setting = pkh.load_index(
                index=step * index_skip,
                all_indices=all_indices,
                data_file=scenes_file,
            )
            temperatures_list.append(setting.temperature)
    temperatures = np.array(temperatures_list)
    return np.array([np.min(temperatures), np.max(temperatures)])


def get_t_data(t_scale: np.ndarray) -> ColorbarSettings:
    # magma plasma cool coolwarm
    lim_small = 0.2
    lim_big = 10

    if t_scale[0] > -lim_small and t_scale[1] < lim_small:
        return ColorbarSettings(vmin=-lim_small, vmax=lim_small, cmap=plt.cm.cool)  # coolwarm
    return ColorbarSettings(vmin=-lim_big, vmax=lim_big, cmap=plt.cm.magma)


def plot_colorbar(fig, axs, cbar_settings):
    for axes in axs:
        position = axes.get_position()
        if position.p0[0] > 0.1:
            position.p0[0] *= 0.8
        position.p1[0] *= 0.9
        axes.set_position(position)

    axes = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    set_ax(axes)
    cbar = fig.colorbar(mappable=cbar_settings.mappable, cax=axes)
    cbar.outline.set_edgecolor("w")
    cbar.outline.set_linewidth(0.2)


def set_ax(axes):
    for spine in axes.spines.values():
        spine.set_edgecolor("w")
        spine.set_linewidth(0.2)
    axes.tick_params(color="w", labelcolor="w", width=0.3, labelsize=5)


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
    plt.savefig(path, **savefig_args, format=extension, dpi=DPI)  # , bbox_inches="tight"
    plt.close()


@dataclass
class AnimationArgs:
    fig: Figure
    time_skip: float
    index_skip: int
    all_indices: List[int]
    scenes_file: BufferedReader
    base_all_indices: Optional[List[int]]
    base_scenes_file: Optional[BufferedReader]
    animation_tqdm: tqdm.tqdm


def make_animation(get_axs, plot_frame, t_scale):
    def animate(step: int, args: AnimationArgs):
        args.animation_tqdm.update(1)
        args.fig.clf()
        axs = get_axs(args.fig)
        scene = pkh.load_index(
            index=step * args.index_skip,
            all_indices=args.all_indices,
            data_file=args.scenes_file,
        )

        if args.base_scenes_file is not None:
            base_scene = pkh.load_index(
                index=step * args.index_skip,
                all_indices=args.base_all_indices,
                data_file=args.base_scenes_file,
            )
        else:
            base_scene = None

        plot_frame(
            axs=axs,
            fig=args.fig,
            scene=scene,
            current_time=step * args.time_skip,
            t_scale=t_scale,
            base_scene=base_scene,
        )
        return args.fig

    return animate


@dataclass
class PlotAnimationConfig:
    save_path: str
    time_skip: float
    index_skip: int
    plot_scenes_count: int
    all_scenes_path: str
    all_calc_scenes_path: Optional[str]


def plot_animation(
    animate: Callable, fig: Figure, config: Config, plot_config: PlotAnimationConfig
):
    fps = int(1 / plot_config.time_skip)
    animation_tqdm = cmh.get_tqdm(
        iterable=range(plot_config.plot_scenes_count + 1),
        config=config,
        desc="Generating animation",
    )

    all_indices = pkh.get_all_indices(plot_config.all_scenes_path)
    scenes_file = pkh.open_file_read(plot_config.all_scenes_path)
    base_all_indices = (
        None
        if plot_config.all_calc_scenes_path is None
        else pkh.get_all_indices(plot_config.all_calc_scenes_path)
    )
    base_scenes_file = (
        None
        if plot_config.all_calc_scenes_path is None
        else pkh.open_file_read(plot_config.all_calc_scenes_path)
    )
    with scenes_file:
        args = AnimationArgs(
            fig=fig,
            time_skip=plot_config.time_skip,
            index_skip=plot_config.index_skip,
            all_indices=all_indices,
            scenes_file=scenes_file,
            base_all_indices=base_all_indices,
            base_scenes_file=base_scenes_file,
            animation_tqdm=animation_tqdm,
        )
        ani = animation.FuncAnimation(
            fig, animate, fargs=(args,), frames=plot_config.plot_scenes_count
        )
        ani.save(plot_config.save_path, writer=None, fps=fps, dpi=DPI, savefig_kwargs=savefig_args)
    plt.close()


def get_frame_annotation(setting, current_time):
    return f"""time: {str(round(current_time, 1))}
nodes: {str(setting.nodes_count)}"""
