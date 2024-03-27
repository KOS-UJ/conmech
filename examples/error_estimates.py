import numba
import numpy as np
from scipy import interpolate
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.tri as tri
import matplotlib.pylab as pl
from conmech.state.state import TemperatureState

# TODO #99


def compare(ref: TemperatureState, sol: TemperatureState):
    ut = 0
    tt = 0
    x = sol.body.mesh.nodes[:, 0]
    y = sol.body.mesh.nodes[:, 1]

    soltri = tri.Triangulation(x, y, triangles=sol.body.mesh.elements)
    u1hi = tri.LinearTriInterpolator(soltri, sol.displacement[:, 0])
    u2hi = tri.LinearTriInterpolator(soltri, sol.displacement[:, 1])
    # thi = tri.LinearTriInterpolator(soltri, sol.temperature)

    for element in ref.body.mesh.elements:
        x0 = ref.body.mesh.nodes[element[0]]
        x1 = ref.body.mesh.nodes[element[1]]
        x2 = ref.body.mesh.nodes[element[2]]
        u1_0 = ref.displacement[element[0], 0]
        u1_1 = ref.displacement[element[1], 0]
        u1_2 = ref.displacement[element[2], 0]
        u1 = u1_0 + u1_1 + u1_2
        u2_0 = ref.displacement[element[0], 1]
        u2_1 = ref.displacement[element[1], 1]
        u2_2 = ref.displacement[element[2], 1]
        u2 = u2_0 + u2_1 + u2_2
        # t0 = ref.temperature[element[0]]
        # t1 = ref.temperature[element[1]]
        # t2 = ref.temperature[element[2]]
        # t = t0 + t1 + t2

        u1dx, u1dy = calculate_dx_dy(x0, u1_0, x1, u1_1, x2, u1_2)
        u2dx, u2dy = calculate_dx_dy(x0, u2_0, x1, u2_1, x2, u2_2)
        # tdx, tdy = calculate_dx_dy(x0, t0, x1, t1, x2, t2)
        u1hdx, u1hdy = calculate_dx_dy(
            x0, u1hi(*x0).compressed(), x1, u1hi(*x1).compressed(), x2, u1hi(*x2).compressed()
        )
        u2hdx, u2hdy = calculate_dx_dy(
            x0, u2hi(*x0).compressed(), x1, u2hi(*x1).compressed(), x2, u2hi(*x2).compressed()
        )
        # thdx, thdy = calculate_dx_dy(x0, thi(*x0), x1, thi(*x1), x2, thi(*x2))

        u1h = u1hi(*x0) + u1hi(*x1) + u1hi(*x2)
        u2h = u2hi(*x0) + u2hi(*x1) + u2hi(*x2)
        ut += (
            (u1 - u1h) ** 2
            + (u2 - u2h) ** 2
            + (u1dx - u1hdx) ** 2
            + (u2dy - u2hdy) ** 2
            + (u1dy - u1hdy + u2dx - u2hdx) ** 2
        ) ** 0.5

        # th = thi(*x0) + thi(*x1) + thi(*x2)
        # tt += ((t - th) ** 2 + (tdx - thdx) ** 2 + (tdy - thdy) ** 2) ** 0.5
    return ut, tt


@numba.njit()
def calculate_dx_dy(x0, u0, x1, u1, x2, u2):
    a1 = x1[0] - x0[0]
    b1 = x1[1] - x0[1]
    c1 = u1 - u0
    a2 = x2[0] - x0[0]
    b2 = x2[1] - x0[1]
    c2 = u2 - u0
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    dx = a / c
    dy = b / c
    return dx, dy


def error_estimates(ref, *args):
    reference_k_h = (9, 6)
    denominator = 1  # TODO 2 ** reference_k_h[0] * 2 ** reference_k_h[1] * 4

    # T = 1
    # kn = 10
    # hn = 6
    # ks = range(kn)
    # hs = range(hn)

    with open(ref, "rb") as output:
        reference = pickle.load(output)
        denominator = len(reference.body.mesh.elements)
        print(f"{denominator=}")

    ue = {}
    # te = np.empty((kn, hn))
    for arg in args:
        with open(arg, "rb") as output:
            solution = pickle.load(output)
            u, t = compare(reference, solution)
            ue[arg] = u / denominator
            # te[k, h] = t / denominator
            # print(k, h, u, t)
    return ue
    # print(repr(ue))

    # print(repr(te))

    # h_ticks = [1 / 2 ** h for h in hs]
    # plt.style.use("seaborn")
    # sns.set(rc={"axes.facecolor": "#E6EDF4"})
    #
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("Velocity error")
    # plt.xlabel("spatial step: h")
    # plt.ylabel("error: $||u-u^h||_V$")
    # optimal = 2 * ue[ks[-1], 1] * np.asarray(h_ticks) ** 1
    # plt.plot(h_ticks, optimal, color="silver", linewidth=4.0, label=None)
    # colors = pl.cm.jet(np.linspace(0, 1, len(ks)))
    # for i, k in enumerate(ks):
    #     plt.plot(
    #         h_ticks, ue[k, hs[0]: hs[-1] + 1], "s-", label="$2^{-" + str(k) + "}$", color=colors[i]
    #     )
    # plt.xticks(h_ticks, h_ticks)
    # plt.legend(title="time step: k")
    # margin = 0.15
    # plt.xlim(max(h_ticks) * (1 + margin), min(h_ticks) * (1 - margin))
    # plt.savefig(
    #     "output/error/displacement.pdf",
    #     transparent=False,
    #     bbox_inches="tight",
    #     format="pdf",
    #     pad_inches=0.1,
    #     dpi=800,
    # )
    # plt.show()

    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("Temperature error")
    # plt.xlabel("spatial step: h")
    # plt.ylabel(r"error: $||\theta-\theta^h||_E$")
    # plt.plot(
    #     h_ticks,
    #     (2 * te[ks[-1], 1] * np.asarray(h_ticks) ** 1),
    #     color="silver",
    #     linewidth=4.0,
    #     label=None,
    # )
    # colors = pl.cm.jet(np.linspace(0, 1, len(ks)))
    # for i, k in enumerate(ks):
    #     plt.plot(
    #         h_ticks, te[k, hs[0]: hs[-1] + 1], "s-", label="$2^{-" + str(k) + "}$", color=colors[i]
    #     )
    # plt.xticks(h_ticks, h_ticks)
    # plt.legend(title="time step: k")
    # margin = 0.15
    # plt.xlim(max(h_ticks) * (1 + margin), min(h_ticks) * (1 - margin))
    # plt.savefig(
    #     "output/error/temperature.pdf",
    #     transparent=False,
    #     bbox_inches="tight",
    #     format="pdf",
    #     pad_inches=0.1,
    #     dpi=800,
    # )
    # plt.show()


if __name__ == "__main__":
    pass
