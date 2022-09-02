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
    ks = range(kn)
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
    # ue = np.asarray([[0.18822859, 0.05856562, 0.03364681, 0.02754594, 0.0222198],
    #        [0.19246518, 0.04755538, 0.02042398, 0.01395414, 0.0104574],
    #        [0.1982109, 0.04473886, 0.0193025, 0.00929889, 0.00495618],
    #        [0.19453114, 0.04524889, 0.0099461, 0.00729312, 0.00306819],
    #        [0.19293092, 0.04545879, 0.01083102, 0.00413664, 0.00229288]])
    # te = np.asarray([[0.0531951, 0.01755566, 0.00905773, 0.00693817, 0.00640698],
    #        [0.05372278, 0.0145119, 0.0053641, 0.00318762, 0.00264197],
    #        [0.05501253, 0.01335008, 0.00406745, 0.00182335, 0.0012826],
    #        [0.05375006, 0.01257149, 0.00331705, 0.00118129, 0.00063839],
    #        [0.05318055, 0.01239222, 0.00316397, 0.00090595, 0.00029507]])

    h_ticks = [1/2**h for h in hs]
    plt.style.use("seaborn")
    sns.set(rc={'axes.facecolor': '#E6EDF4'})

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Velocity error")
    plt.xlabel("spatial step: h")
    plt.ylabel("error: $||v-v^h||_V$")
    plt.plot(h_ticks, (ue[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0, label=f"optimal")
    for k in ks:
        plt.plot(h_ticks, ue[k, :], "s-", label="$2^{-" + str(k) + "}$")
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    plt.xlim(max(h_ticks)*1.125, min(h_ticks)*0.875)
    plt.show()

    plt.xscale("log")
    plt.yscale("log")
    plt.title("Temperature error")
    plt.xlabel("spatial step: h")
    plt.ylabel(r"error: $||\theta-\theta^h||_V$")
    plt.plot(h_ticks, (te[ks[0], 0] ** .5 * np.asarray(h_ticks)) ** 2, color="silver", linewidth=4.0, label=f"optimal")
    for k in ks:
        plt.plot(h_ticks, te[k, :], "s-", label="$2^{-" + str(k) + "}$")
    plt.xticks(h_ticks, h_ticks)
    plt.legend(title="time step: k")
    plt.xlim(max(h_ticks)*1.125, min(h_ticks)*0.875)
    plt.show()

