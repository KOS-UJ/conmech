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
    reference_k_h = (5, 5)
    denominator = 2**reference_k_h[0] * 2**reference_k_h[1] * 4

    with open(f'output/temp/k_{reference_k_h[0]}_h_{reference_k_h[1]}', 'rb') as output:
        reference = pickle.load(output)

    T = 1
    kn = 5
    hn = 5
    ks = range(1, kn)
    hs = range(hn)
    ue = np.empty((kn, hn))
    te = np.empty((kn, hn))
    for h in hs:
        for k in ks:
            with open(f'output/temp/k_{k}_h_{h}', 'rb') as output:
                solution = pickle.load(output)
                u, t = compare(reference, solution)
                ue[k, h] = u / denominator
                te[k, h] = t / denominator
                print(k, h, u, t)
    print(repr(ue))
    print(repr(te))
    # ue = np.asarray([[2.71252237e-316, 0.00000000e+000, 3.47852406e-308,
    #     1.69138275e-258, 4.59714141e-287],
    #    [1.92465185e-001, 4.75553786e-002, 2.04239809e-002,
    #     1.39541358e-002, 1.34573983e-002],
    #    [1.98210898e-001, 4.47388630e-002, 1.93024976e-002,
    #     9.29889060e-003, 1.09561817e-002],
    #    [1.94531144e-001, 4.52488889e-002, 9.94610202e-003,
    #     7.29312294e-003, 8.06819093e-003],
    #    [1.92930918e-001, 4.54587880e-002, 1.08310176e-002,
    #     4.13663850e-003, 3.99287780e-003]])
    # te = np.asarray([[2.25167296e-316, 0.00000000e+000, 7.72485098e+228,
    #     2.71248126e-316, 5.98129656e-154],
    #    [5.37227785e-002, 1.45118972e-002, 5.36409957e-003,
    #     3.18761666e-003, 2.64197124e-003],
    #    [5.50125299e-002, 1.33500827e-002, 4.06744521e-003,
    #     1.82334603e-003, 1.28260484e-003],
    #    [5.37500599e-002, 1.25714895e-002, 3.31705026e-003,
    #     1.18128980e-003, 6.38392841e-004],
    #    [5.31805489e-002, 1.23922240e-002, 3.16397024e-003,
    #     9.05953308e-004, 2.95070851e-004]])

    h_ticks = [1/2**h for h in hs]
    plt.style.use("seaborn")
    sns.set(rc={'axes.facecolor': '#E6EDF4'})

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Velocity error")
    plt.xlabel("spatial step: h")
    plt.ylabel("error: $||v-v^h||_V$")
    plt.plot(h_ticks, (ue[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0)
    for k in ks:
        plt.plot(h_ticks, ue[k, :], "s-", label="some")
    plt.xticks(h_ticks, h_ticks)
    plt.show()

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Temperature error")
    plt.xlabel("spatial step: h")
    plt.ylabel(r"error: $||\theta-\theta^h||_V$")
    plt.plot(h_ticks, (te[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0)
    for k in ks:
        plt.plot(h_ticks, te[k, :], "s-")
    plt.xticks(h_ticks, h_ticks)
    plt.show()

