import matplotlib.pyplot as plt
import numpy as np


def draw_VI_vs_HVI():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    LW = 4

    x_center = np.linspace(-0.5, 0.5, 500)
    x_left = np.linspace(-1, -0.5, 500)
    x_right = np.linspace(0.5, 1, 500)

    j_nc_center = 4 * x_center**2
    j_nc_left = x_left**2 + 0.75
    j_nc_right = x_right**2 + 0.75

    NONCONVEX = 1
    axs[0, NONCONVEX].plot(x_left, j_nc_left, color="#000080", lw=LW)
    axs[0, NONCONVEX].plot(x_center, j_nc_center, color="#000080", lw=LW)
    axs[0, NONCONVEX].plot(x_right, j_nc_right, color="#000080", lw=LW)
    axs[0, NONCONVEX].text(0.7, 1.9, r"$j(x)$", fontsize=26, color="#000080")

    dj_nc_center = 8 * x_center
    dj_nc_left = 2 * x_left
    dj_nc_right = 2 * x_right

    axs[1, NONCONVEX].plot(x_left, dj_nc_left, color="#4169E1", lw=LW)
    axs[1, NONCONVEX].plot(x_center, dj_nc_center, color="#4169E1", lw=LW)
    axs[1, NONCONVEX].plot(x_right, dj_nc_right, color="#4169E1", lw=LW)
    axs[1, NONCONVEX].vlines(-0.5, -4, -1, color="#4169E1", lw=LW, linestyle="-")
    axs[1, NONCONVEX].vlines(0.5, 1, 4, color="#4169E1", lw=LW, linestyle="-")
    axs[1, NONCONVEX].text(0.7, 2.6, r"$\partial j(x)$", fontsize=26, color="#4169E1")

    j_c_center = x_center**2
    j_c_left = 4 * x_left**2 - 0.75
    j_c_right = 4 * x_right**2 - 0.75

    CONVEX = (NONCONVEX + 1) % 2
    axs[0, CONVEX].plot(x_left, j_c_left, color="#8B0000", lw=LW)
    axs[0, CONVEX].plot(x_center, j_c_center, color="#8B0000", lw=LW)
    axs[0, CONVEX].plot(x_right, j_c_right, color="#8B0000", lw=LW)
    axs[0, CONVEX].text(0.7, 1.0, r"$j(x)$", fontsize=26, color="#8B0000")

    dj_c_center = 2 * x_center
    dj_c_left = 8 * x_left
    dj_c_right = 8 * x_right

    axs[1, CONVEX].plot(x_left, dj_c_left, color="#CD5C5C", lw=LW)
    axs[1, CONVEX].plot(x_center, dj_c_center, color="#CD5C5C", lw=LW)
    axs[1, CONVEX].plot(x_right, dj_c_right, color="#CD5C5C", lw=LW)
    axs[1, CONVEX].vlines(-0.5, -4, -1, color="#CD5C5C", lw=LW, linestyle="-")
    axs[1, CONVEX].vlines(0.5, 1, 4, color="#CD5C5C", lw=LW, linestyle="-")
    axs[1, CONVEX].text(0.7, 4.0, r"$\partial j(x)$", fontsize=26, color="#CD5C5C")

    for i in range(2):
        for j in range(2):
            axs[i, j].spines["top"].set_visible(False)
            axs[i, j].spines["right"].set_visible(False)
            axs[i, j].spines["left"].set_position("zero")
            axs[i, j].spines["bottom"].set_position("zero")

            axs[i, j].axvline(-0.5, color="gray", linestyle="-.", lw=1, alpha=0.5)
            axs[i, j].axvline(0.5, color="gray", linestyle="-.", lw=1, alpha=0.5)

            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

            axs[i, j].set_xlim(-1.2, 1.2)
            if i == 0:
                axs[i, j].set_ylim(-0.2, 3.5)
            else:
                axs[i, j].set_ylim(-9.0, 9.0)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_VI_vs_HVI()
