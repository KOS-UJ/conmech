import numba
import numpy as np
from scipy import interpolate
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import matplotlib.tri as tri

from conmech.state.state import State, TemperatureState


def compare(ref: TemperatureState, sol: TemperatureState):
    ut = 0
    tt = 0
    x = sol.body.mesh.initial_nodes[:, 0]
    y = sol.body.mesh.initial_nodes[:, 1]

    soltri = tri.Triangulation(x, y, triangles=sol.body.mesh.elements)
    u1hi = tri.LinearTriInterpolator(soltri, sol.displacement[:, 0])
    u2hi = tri.LinearTriInterpolator(soltri, sol.displacement[:, 1])
    thi = tri.LinearTriInterpolator(soltri, sol.temperature)

    for element in ref.body.mesh.elements:
        x0 = ref.body.mesh.initial_nodes[element[0]]
        x1 = ref.body.mesh.initial_nodes[element[1]]
        x2 = ref.body.mesh.initial_nodes[element[2]]
        u1_0 = ref.displacement[element[0], 0]
        u1_1 = ref.displacement[element[1], 0]
        u1_2 = ref.displacement[element[2], 0]
        u1 = u1_0 + u1_1 + u1_2
        u2_0 = ref.displacement[element[0], 1]
        u2_1 = ref.displacement[element[1], 1]
        u2_2 = ref.displacement[element[2], 1]
        u2 = u2_0 + u2_1 + u2_2
        t0 = ref.temperature[element[0]]
        t1 = ref.temperature[element[1]]
        t2 = ref.temperature[element[2]]
        t = t0 + t1 + t2

        u1dx, u1dy = calculate_dx_dy(x0, u1_0, x1, u1_1, x2, u1_2)
        u2dx, u2dy = calculate_dx_dy(x0, u2_0, x1, u2_1, x2, u2_2)
        tdx, tdy = calculate_dx_dy(x0, t0, x1, t1, x2, t2)
        u1hdx, u1hdy = calculate_dx_dy(x0, u1hi(*x0), x1, u1hi(*x1), x2, u1hi(*x2))
        u2hdx, u2hdy = calculate_dx_dy(x0, u2hi(*x0), x1, u2hi(*x1), x2, u2hi(*x2))
        thdx, thdy = calculate_dx_dy(x0, thi(*x0), x1, thi(*x1), x2, thi(*x2))


        u1h = u1hi(*x0) + u1hi(*x1) + u1hi(*x2)
        u2h = u2hi(*x0) + u2hi(*x1) + u2hi(*x2)
        ut += (u1 - u1h)**2 + (u2 - u2h)**2 \
              + (u1dx - u1hdx)**2 + (u2dx - u2hdx)**2 \
              + (u1dy - u1hdy)**2 + (u2dy - u2hdy)**2

        th = thi(*x0) + thi(*x1) + thi(*x2)
        tt += (t - th)**2 + (tdx - thdx)**2 + (tdy - thdy)**2
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


if __name__ == "__main__":
    reference_k_h = (9, 6)
    denominator = 2 ** reference_k_h[0] * 2 ** reference_k_h[1] * 4

    T = 1
    kn = 10
    hn = 6
    ks = range(kn)
    hs = range(hn)

    # with open(f'output/temp/k_{reference_k_h[0]}_h_{reference_k_h[1]}', 'rb') as output:
    #     reference = pickle.load(output)
    #
    # ue = np.empty((kn, hn))
    # te = np.empty((kn, hn))
    # for h in hs:
    #     for k in ks:
    #         with open(f'output/temp/k_{k}_h_{h}', 'rb') as output:
    #             solution = pickle.load(output)
    #             u, t = compare(reference, solution)
    #             ue[k, h] = u / denominator
    #             te[k, h] = t / denominator
    #             print(k, h, u, t)
    # print(repr(ue))
    # print(repr(te))

    ue = np.array([[0.02392508, 0.00757334, 0.00461858, 0.00387066, 0.00372696,
        0.00369577],
       [0.02445688, 0.00613711, 0.00294297, 0.0022364 , 0.00218759,
        0.00205   ],
       [0.02517737, 0.00563762, 0.00268352, 0.00165538, 0.00134046,
        0.00108176],
       [0.02471674, 0.00542868, 0.00219769, 0.00149107, 0.00127985,
        0.00109518],
       [0.02414908, 0.00562159, 0.00173276, 0.00096075, 0.00076951,
        0.00064924],
       [0.02322422, 0.00550854, 0.00160304, 0.00068173, 0.00054045,
        0.00046991],
       [0.02445768, 0.00538306, 0.00148226, 0.00054621, 0.00028901,
        0.00025098],
       [0.02413768, 0.00657959, 0.00164731, 0.00052562, 0.00023513,
        0.00014581],
       [0.02571625, 0.00548802, 0.00150959, 0.00042123, 0.00021417,
        0.00011267],
       [0.02548665, 0.00551344, 0.0013225, 0.00039182, 0.00013362,
        0.000077],
        ])
    te = np.array([[6.68840948e-03, 2.23317518e-03, 1.17099191e-03, 9.06050167e-04,
        8.39989673e-04, 8.22465467e-04],
       [6.75441422e-03, 1.85291994e-03, 7.09618331e-04, 4.38009511e-04,
        3.69678863e-04, 3.52221612e-04],
       [6.91566725e-03, 1.70744362e-03, 5.47374414e-04, 2.67321760e-04,
        1.99243888e-04, 1.82806564e-04],
       [6.75787377e-03, 1.63120169e-03, 4.76841744e-04, 1.87150791e-04,
        1.18782585e-04, 1.04276345e-04],
       [6.59880056e-03, 1.58029758e-03, 4.37482209e-04, 1.53597369e-04,
        7.99389647e-05, 6.24184358e-05],
       [6.32669643e-03, 1.56423996e-03, 4.22772740e-04, 1.29527099e-04,
        4.33278681e-05, 4.05234908e-05],
       [6.64328975e-03, 1.56851000e-03, 4.05988264e-04, 1.14360839e-04,
        3.99946472e-05, 2.76104294e-05],
       [7.00301325e-03, 1.56599552e-03, 3.89068709e-04, 1.03008683e-04,
        2.75967602e-05, 1.42770267e-05],
       [7.06181587e-03, 1.53872700e-03, 3.95523813e-04, 1.05068146e-04,
        2.04475482e-05, 4.67294342e-06],
       [6.64730259e-03, 1.53906169e-03, 3.74747159e-04, 9.02876000e-05,
        1.81302270e-05, 4.26193823e-06]])

    h_ticks = [1 / 2 ** h for h in hs]
    plt.style.use("seaborn")
    sns.set(rc={'axes.facecolor': '#E6EDF4'})

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Velocity error")
    plt.xlabel("spatial step: h")
    plt.ylabel("error: $||v-v^h||_V$")
    optimal = (ue[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2
    plt.plot(h_ticks, optimal, color="silver", linewidth=4.0, label="optimal")
    for k in ks:
        plt.plot(h_ticks, ue[k, hs[0]:hs[-1]+1], "s-", label="$2^{-" + str(k) + "}$")
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    plt.xlim(max(h_ticks)*1.125, min(h_ticks)*0.875)
    plt.savefig(
        "output/error/velocity.pdf",
        transparent=False,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.1,
        dpi=800,
    )
    plt.show()

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Temperature error")
    plt.xlabel("spatial step: h")
    plt.ylabel(r"error: $||\theta-\theta^h||_E$")
    plt.plot(h_ticks, (te[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0, label="optimal")
    for k in ks:
        plt.plot(h_ticks, te[k, hs[0]:hs[-1]+1], "s-", label="$2^{-" + str(k) + "}$")
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    plt.xlim(max(h_ticks)*1.125, min(h_ticks)*0.875)
    plt.savefig(
        "output/error/temperature.pdf",
        transparent=False,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.1,
        dpi=800,
    )
    plt.show()