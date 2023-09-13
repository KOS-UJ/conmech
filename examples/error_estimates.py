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
    thi = tri.LinearTriInterpolator(soltri, sol.temperature)

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
        ut += (
            (u1 - u1h) ** 2
            + (u2 - u2h) ** 2
            + (u1dx - u1hdx) ** 2
            + (u2dy - u2hdy) ** 2
            + (u1dy - u1hdy + u2dx - u2hdx) ** 2
        ) ** 0.5

        th = thi(*x0) + thi(*x1) + thi(*x2)
        tt += ((t - th) ** 2 + (tdx - thdx) ** 2 + (tdy - thdy) ** 2) ** 0.5
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

    # ue = np.asarray([[3.51064292e-03, 6.60816772e-04, 1.57858401e-04, 8.49138212e-05,
    #         7.25934268e-05, 7.06925935e-05],
    #        [3.61655733e-03, 6.48988624e-04, 1.15492804e-04, 3.93501894e-05,
    #         2.77189199e-05, 2.61284821e-05],
    #        [3.76727797e-03, 6.54611297e-04, 1.12705825e-04, 2.88279473e-05,
    #         1.84614324e-05, 1.72210597e-05],
    #        [3.67165437e-03, 6.50281211e-04, 1.04942153e-04, 2.40933567e-05,
    #         1.22119143e-05, 9.96705359e-06],
    #        [3.55872410e-03, 6.90665486e-04, 9.86944361e-05, 1.94383337e-05,
    #         6.82798321e-06, 4.58554490e-06],
    #        [3.38251554e-03, 6.68536915e-04, 9.80099335e-05, 1.72279032e-05,
    #         5.17652041e-06, 1.90904509e-06],
    #        [3.63972474e-03, 6.55831166e-04, 9.67532875e-05, 1.64753487e-05,
    #         3.39027433e-06, 1.14619503e-06],
    #        [3.84269955e-03, 6.57062689e-04, 9.43591112e-05, 1.60484103e-05,
    #         3.06911182e-06, 6.50277890e-07],
    #        [3.89351253e-03, 6.53322099e-04, 9.78647821e-05, 2.84040210e-05,
    #         3.21183711e-06, 6.66848485e-07],
    #        [3.56222413e-03, 7.58684791e-04, 1.01876870e-04, 1.69102222e-05,
    #         3.16765328e-06, 6.76218102e-07]])
    # te = np.asarray([[2.51983950e-04, 3.65203893e-05, 1.40350369e-05, 9.03639485e-06,
    #         7.95321197e-06, 7.68786449e-06],
    #        [2.58693772e-04, 2.74905815e-05, 7.45569008e-06, 3.86424203e-06,
    #         3.13708124e-06, 2.96597709e-06],
    #        [2.71314633e-04, 2.36231095e-05, 4.67394194e-06, 1.76615566e-06,
    #         1.24542801e-06, 1.13771908e-06],
    #        [2.62131942e-04, 2.20884726e-05, 3.55819379e-06, 9.05536091e-07,
    #         4.66154672e-07, 3.82321131e-07],
    #        [2.53106093e-04, 2.17287469e-05, 3.17266640e-06, 6.22052400e-07,
    #         2.10371400e-07, 1.25786283e-07],
    #        [2.37256953e-04, 2.13680291e-05, 3.10738659e-06, 5.26606195e-07,
    #         1.20051134e-07, 5.10547779e-08],
    #        [2.56370262e-04, 2.11524557e-05, 3.00687181e-06, 4.92886048e-07,
    #         1.06324965e-07, 2.91076083e-08],
    #        [2.79046413e-04, 2.09837016e-05, 3.00190538e-06, 4.83016121e-07,
    #         9.65938350e-08, 2.08012631e-08],
    #        [2.83269851e-04, 2.13542387e-05, 3.13429246e-06, 6.21141133e-07,
    #         9.40967476e-08, 1.82546450e-08],
    #        [2.54791829e-04, 2.08020592e-05, 2.96826861e-06, 4.77383581e-07,
    #         9.37414078e-08, 1.81747962e-08]])
    # array([[3.14681433e-03, 5.78708271e-04, 1.42346139e-04, 7.61790675e-05,
    #         6.42927759e-05, 6.19299357e-05],
    #        [3.25632082e-03, 5.39096000e-04, 9.65981575e-05, 3.50823920e-05,
    #         2.57918955e-05, 2.45186524e-05],
    #        [3.41123943e-03, 5.28468199e-04, 9.15348410e-05, 2.56499495e-05,
    #         1.81253651e-05, 1.72970692e-05],
    #        [3.31525800e-03, 5.17078003e-04, 8.21106003e-05, 2.06276332e-05,
    #         1.18540371e-05, 1.01068609e-05],
    #        [3.19620516e-03, 5.15337705e-04, 7.44393552e-05, 1.56228857e-05,
    #         6.31159301e-06, 4.60707382e-06],
    #        [3.01144473e-03, 5.09017138e-04, 7.29054520e-05, 1.30637947e-05,
    #         4.26555682e-06, 1.84300202e-06],
    #        [3.26973059e-03, 5.08055560e-04, 7.11474149e-05, 1.21325368e-05,
    #         2.61712020e-06, 1.03775258e-06],
    #        [3.48839816e-03, 5.15453847e-04, 6.74653739e-05, 1.14800101e-05,
    #         2.28290731e-06, 5.13143518e-07],
    #        [3.53998890e-03, 5.05289847e-04, 7.01592614e-05, 1.97090876e-05,
    #         2.37363292e-06, 5.21670964e-07],
    #        [3.20002421e-03, 5.59609645e-04, 7.12836450e-05, 1.17558040e-05,
    #         2.35212189e-06, 5.25712091e-07]])
    # array([[2.51983950e-04, 3.65203893e-05, 1.40350369e-05, 9.03639485e-06,
    #         7.95321197e-06, 7.68786449e-06],
    #        [2.58693772e-04, 2.74905815e-05, 7.45569008e-06, 3.86424203e-06,
    #         3.13708124e-06, 2.96597709e-06],
    #        [2.71314633e-04, 2.36231095e-05, 4.67394194e-06, 1.76615566e-06,
    #         1.24542801e-06, 1.13771908e-06],
    #        [2.62131942e-04, 2.20884726e-05, 3.55819379e-06, 9.05536091e-07,
    #         4.66154672e-07, 3.82321131e-07],
    #        [2.53106093e-04, 2.17287469e-05, 3.17266640e-06, 6.22052400e-07,
    #         2.10371400e-07, 1.25786283e-07],
    #        [2.37256953e-04, 2.13680291e-05, 3.10738659e-06, 5.26606195e-07,
    #         1.20051134e-07, 5.10547779e-08],
    #        [2.56370262e-04, 2.11524557e-05, 3.00687181e-06, 4.92886048e-07,
    #         1.06324965e-07, 2.91076083e-08],
    #        [2.79046413e-04, 2.09837016e-05, 3.00190538e-06, 4.83016121e-07,
    #         9.65938350e-08, 2.08012631e-08],
    #        [2.83269851e-04, 2.13542387e-05, 3.13429246e-06, 6.21141133e-07,
    #         9.40967476e-08, 1.82546450e-08],
    #        [2.54791829e-04, 2.08020592e-05, 2.96826861e-06, 4.77383581e-07,
    #         9.37414078e-08, 1.81747962e-08]])
    ue = np.asarray(
        [
            [0.02243367, 0.00945494, 0.0047539, 0.00355776, 0.00331022, 0.00324383],
            [0.02291673, 0.00892371, 0.00371959, 0.00238767, 0.00211572, 0.00208179],
            [0.02358811, 0.008724, 0.0035803, 0.00198949, 0.00178018, 0.00175999],
            [0.02318723, 0.0086035, 0.00331536, 0.00177648, 0.00144094, 0.0013479],
            [0.02264709, 0.00844671, 0.00308759, 0.0014665, 0.0010232, 0.00091025],
            [0.02179787, 0.00843854, 0.00304546, 0.00128954, 0.0008246, 0.00056634],
            [0.02290356, 0.00848389, 0.00299967, 0.00122134, 0.00058402, 0.00041183],
            [0.023919, 0.00860834, 0.00283703, 0.00116145, 0.00051856, 0.00026714],
            [0.02413345, 0.00849438, 0.00292525, 0.00167922, 0.00054691, 0.00027684],
            [0.0226687, 0.00899269, 0.00298859, 0.00119544, 0.00054671, 0.00027651],
        ]
    )
    te = np.asarray(
        [
            [
                6.82540994e-03,
                2.56319187e-03,
                1.55330144e-03,
                1.21156666e-03,
                1.12150221e-03,
                1.09839411e-03,
            ],
            [
                6.91144671e-03,
                2.20870880e-03,
                1.11799969e-03,
                7.59382694e-04,
                6.59473669e-04,
                6.33192738e-04,
            ],
            [
                7.08020402e-03,
                2.04717024e-03,
                8.88736589e-04,
                5.11961797e-04,
                4.05334124e-04,
                3.77880307e-04,
            ],
            [
                6.94868359e-03,
                1.98380168e-03,
                7.76518636e-04,
                3.73911777e-04,
                2.53195833e-04,
                2.20284279e-04,
            ],
            [
                6.81657653e-03,
                1.97525209e-03,
                7.31400392e-04,
                3.12309945e-04,
                1.74347377e-04,
                1.29991907e-04,
            ],
            [
                6.58256367e-03,
                1.95825699e-03,
                7.23236071e-04,
                2.85801473e-04,
                1.31119651e-04,
                8.54069481e-05,
            ],
            [
                6.86190756e-03,
                1.94632624e-03,
                7.10184903e-04,
                2.74813899e-04,
                1.23550007e-04,
                6.53353204e-05,
            ],
            [
                7.18011214e-03,
                1.93730438e-03,
                7.08811210e-04,
                2.70437905e-04,
                1.16131191e-04,
                5.46813675e-05,
            ],
            [
                7.23629516e-03,
                1.95936459e-03,
                7.25445385e-04,
                3.02455407e-04,
                1.13763883e-04,
                5.06783627e-05,
            ],
            [
                6.84668307e-03,
                1.92875550e-03,
                7.02538581e-04,
                2.66845080e-04,
                1.13258321e-04,
                5.03505677e-05,
            ],
        ]
    )

    h_ticks = [1 / 2**h for h in hs]
    plt.style.use("seaborn")
    sns.set(rc={"axes.facecolor": "#E6EDF4"})

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Velocity error")
    plt.xlabel("spatial step: h")
    plt.ylabel("error: $||v-v^h||_V$")
    optimal = 2 * ue[ks[-1], 1] * np.asarray(h_ticks) ** 1
    plt.plot(h_ticks, optimal, color="silver", linewidth=4.0, label=None)
    colors = pl.cm.jet(np.linspace(0, 1, len(ks)))
    for i, k in enumerate(ks):
        plt.plot(
            h_ticks, ue[k, hs[0] : hs[-1] + 1], "s-", label="$2^{-" + str(k) + "}$", color=colors[i]
        )
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    margin = 0.15
    plt.xlim(max(h_ticks) * (1 + margin), min(h_ticks) * (1 - margin))
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
    plt.plot(
        h_ticks,
        (2 * te[ks[-1], 1] * np.asarray(h_ticks) ** 1),
        color="silver",
        linewidth=4.0,
        label=None,
    )
    colors = pl.cm.jet(np.linspace(0, 1, len(ks)))
    for i, k in enumerate(ks):
        plt.plot(
            h_ticks, te[k, hs[0] : hs[-1] + 1], "s-", label="$2^{-" + str(k) + "}$", color=colors[i]
        )
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    margin = 0.15
    plt.xlim(max(h_ticks) * (1 + margin), min(h_ticks) * (1 - margin))
    plt.savefig(
        "output/error/temperature.pdf",
        transparent=False,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.1,
        dpi=800,
    )
    plt.show()
