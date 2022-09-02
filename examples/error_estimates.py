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
    u1hi = tri.LinearTriInterpolator(soltri, sol.velocity[:, 0])
    u2hi = tri.LinearTriInterpolator(soltri, sol.velocity[:, 1])
    thi = tri.LinearTriInterpolator(soltri, sol.temperature)

    for element in ref.body.mesh.elements:
        x0 = ref.body.mesh.initial_nodes[element[0]]
        x1 = ref.body.mesh.initial_nodes[element[1]]
        x2 = ref.body.mesh.initial_nodes[element[2]]
        u1_0 = ref.velocity[element[0], 0]
        u1_1 = ref.velocity[element[1], 0]
        u1_2 = ref.velocity[element[2], 0]
        u1 = u1_0 + u1_1 + u1_2
        u2_0 = ref.velocity[element[0], 1]
        u2_1 = ref.velocity[element[1], 1]
        u2_2 = ref.velocity[element[2], 1]
        u2 = u2_0 + u2_1 + u2_2
        t0 = ref.temperature[element[0]]
        t1 = ref.temperature[element[1]]
        t2 = ref.temperature[element[2]]
        t = t0 + t1 + t2

        u1h = u1hi(*x0) + u1hi(*x1) + u1hi(*x2)
        u2h = u2hi(*x0) + u2hi(*x1) + u2hi(*x2)
        ut += np.linalg.norm(u1 - u1h) + np.linalg.norm(u2 - u2h)
        th = thi(*x0) + thi(*x1) + thi(*x2)
        tt += np.linalg.norm(t - th)
    return ut, tt


if __name__ == "__main__":
    reference_k_h = (9, 6)
    denominator = 2 ** reference_k_h[0] * 2 ** reference_k_h[1] * 4

    # with open(f'output/temp/k_{reference_k_h[0]}_h_{reference_k_h[1]}', 'rb') as output:
    #     reference = pickle.load(output)

    T = 1
    kn = 10
    hn = 6
    ks = range(1, kn)
    hs = range(hn)
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

    ue = np.array([[0.,  0,  0, 0,  0,  0, 0],
       [ 1.77550752e-003,  6.35420955e-003,  7.44978705e-003,
         7.91286458e-003,  8.12439520e-003,  8.22110592e-003],
       [ 2.28661937e-003,  9.54119396e-004,  1.00135751e-003,
         9.66298165e-004,  9.61893012e-004,  9.60498757e-004],
       [ 6.10504339e-003,  1.61315083e-003,  1.84049612e-003,
         2.02823072e-003,  1.99995538e-003,  2.29089141e-003],
       [ 5.74062694e-003,  1.13505440e-003,  2.05072433e-003,
         1.85100847e-003,  2.14714361e-003,  2.09567583e-003],
       [ 8.76212774e-003,  2.93597586e-003,  1.33972505e-003,
         1.27124260e-003,  1.27957335e-003,  1.44561459e-003],
       [ 7.57017364e-003,  1.31018740e-003,  7.64620003e-004,
         6.00388504e-004,  9.22631482e-004,  7.44054621e-004],
       [ 2.69099319e-003,  4.60718105e-004,  1.03652734e-003,
         5.13930230e-004,  4.02472545e-004,  5.22232419e-004],
       [ 2.92789687e-003,  1.80529092e-003,  2.12118824e-003,
         4.08124849e-003,  2.33230651e-004,  1.99608913e-004],
       [ 1.34588104e-003,  7.81684601e-003,  1.10395708e-003,
         6.66969478e-004,  1.60830293e-004,  4.66301484e-005]])
    te = np.array([[0., 0., 0., 0., 0., 0., 0.],
       [6.75441422e-003, 1.85291994e-003, 7.09618331e-004,
        4.38009511e-004, 3.69678863e-004, 3.52221612e-004],
       [6.91566725e-003, 1.70744362e-003, 5.47374414e-004,
        2.67321760e-004, 1.99243888e-004, 1.82806564e-004],
       [6.75787377e-003, 1.63120169e-003, 4.76841744e-004,
        1.87150791e-004, 1.18782585e-004, 1.04276345e-004],
       [6.59880056e-003, 1.58029758e-003, 4.37482209e-004,
        1.53597369e-004, 7.99389647e-005, 6.24184358e-005],
       [6.32669643e-003, 1.56423996e-003, 4.22772740e-004,
        1.29527099e-004, 4.33278681e-005, 4.05234908e-005],
       [6.64328975e-003, 1.56851000e-003, 4.05988264e-004,
        1.14360839e-004, 3.99946472e-005, 2.76104294e-005],
       [7.00301325e-003, 1.56599552e-003, 3.89068709e-004,
        1.03008683e-004, 2.75967602e-005, 1.42770267e-005],
       [7.06181587e-003, 1.53872700e-003, 3.95523813e-004,
        1.05068146e-004, 2.04475482e-005, 4.67294342e-006],
       [6.64730259e-003, 1.53906169e-003, 3.74747159e-004,
        9.02876000e-005, 1.81302270e-005, 4.26193823e-006]])

    h_ticks = [1 / 2 ** h for h in hs]
    plt.style.use("seaborn")
    sns.set(rc={'axes.facecolor': '#E6EDF4'})

    # plt.xscale("log")
    # plt.yscale("log")
    plt.title("Velocity error")
    plt.xlabel("spatial step: h")
    plt.ylabel("error: $||v-v^h||_V$")
    plt.plot(h_ticks, (ue[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0)
    for k in ks:
        plt.plot(h_ticks, ue[k, :], "s-", label="some")
    plt.xticks(h_ticks, h_ticks)
    plt.show()

    # plt.xscale("log")
    # plt.yscale("log")
    plt.title("Temperature error")
    plt.xlabel("spatial step: h")
    plt.ylabel(r"error: $||\theta-\theta^h||_V$")
    plt.plot(h_ticks, (te[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0)
    for k in ks:
        plt.plot(h_ticks, te[k, :], "s-")
    plt.xticks(h_ticks, h_ticks)
    plt.show()
