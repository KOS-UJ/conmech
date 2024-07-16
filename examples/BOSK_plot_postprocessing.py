import numpy as np
from matplotlib import pyplot as plt

from conmech.plotting.membrane import plot_limit_points
from conmech.state.state import State
from matplotlib.ticker import StrMethodFormatter


def case1():
    states = {}
    for kappa, beta in (
                               (0.0, 0.0), (0.0, 0.25), (0.0, 0.5), (0.0, 0.75),
                               (0.0, 1.0),
                               (1.0, 0.0), (1.0, 0.5), (10.0, 0.5),
                               (100.0, 0.5),
                       )[:]:
        print((kappa, beta))
        states[kappa, beta] = \
            State.load(f"output/BOSK.pub/c1_kappa={kappa:.2f};beta={beta:.2f}")

    c1_reference(states, output_path='output/BOSK.pub')
    c1_steady_state(states, output_path='output/BOSK.pub')
    c1_influ(states, output_path='output/BOSK.pub')


def show(output_path, name):
    plt.gca().yaxis.set_major_formatter(
        StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    plt.gca().xaxis.set_major_formatter(
        StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path + f'/{name}.png', format='png', dpi=800)


def c1_reference(states, output_path=None):
    plt.rc('axes', axisbelow=True)

    kappa = 100.0
    beta = 0.5
    timesteps = (0.875, 2.0)
    labels = ('b)', 'a)')
    eps = 0.001

    all_zeros = []

    for subnumber, timestep in zip(labels, timesteps):
        intersect = states[kappa, beta].products['intersection at 1.00']
        plt.figure(figsize=(5, 4))
        zeros = states[kappa, beta].products['limit points at 1.00'].data[
            timestep]
        all_zeros.extend(zeros)
        plt.xticks((0.0, *zeros, 1.0), rotation=90)
        plt.yticks((0.0, 1.0, 1.8))
        plt.grid()
        plt.axhspan(1.0, 1.8, alpha=0.1, color='lightskyblue')

        for t, v in intersect.data.items():
            if timestep - eps > t or t > timestep + eps:
                continue
            print(t)
            plt.plot(*v, color=f'black')
        states[kappa, beta].products['limit points at 1.00'].range(0.00, 8.00)

        plt.scatter(zeros, np.ones_like(zeros), color=f'black', s=10)
        plt.ylim(0.0, 1.8)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(fr'$\kappa={kappa:.2f}$, $\beta={beta:.2f}$, $t={timestep}$')
        plt.title(subnumber, loc='left')

        show(output_path, name=f'boundary_{timestep:.3f}')

    plt.figure(figsize=(10, 4))
    plot_limit_points(
        states[kappa, beta].products['limit points at 1.00'].range(0.00, 8.00),
        title=None, color=f'black', finish=False)

    plt.xticks((0.0, *timesteps, 8.0))
    plt.yticks((0.0, *all_zeros, 1.0))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("y")
    plt.title(fr'$\kappa={kappa:.2f}$, $\beta={beta:.2f}$')
    plt.title('c)', loc='left')

    show(output_path, name='reference')


def c1_steady_state(states, output_path=None):
    plt.figure(figsize=(10, 4))
    kappa = 0.0
    beta = 0.0
    plot_limit_points(
        states[kappa, beta].products['limit points at 1.00'].range(0.00, 8.00),
        title=None, color=f'gray', finish=False, label=fr'$\beta={beta:.2f}$')
    beta = 1.0
    plot_limit_points(
        states[kappa, beta].products['limit points at 1.00'].range(0.00, 8.00),
        title=None, color=f'salmon', finish=False, label=fr'$\beta={beta:.2f}$')
    plt.xticks((0.0, 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.0))
    plt.yticks((0.0, 1.0))
    plt.xlabel("time")
    plt.ylabel("y")
    plt.title(fr'$\kappa={kappa:.2f}$')

    plt.legend(loc='center right')
    show(output_path, name='steady_state')


def c1_influ(states, output_path=None):
    cases = {('a)', 'kappa'):
                 (('lightskyblue', 100.0, 0.5), ('yellowgreen', 10.0, 0.5),
                  ('gold', 1.0, 0.5), ('salmon', 0.0, 0.5)),
             ('b)', 'beta'):
                 (('lightskyblue', 0.0, 0.25), ('yellowgreen', 0.0, 0.5),
                  ('gold', 0.0, 0.75), ('salmon', 0.0, 1.0))}
    for (subnumber, var), case in cases.items():
        plt.figure(figsize=(6, 4.5))
        for c, kappa, beta in case:
            print((kappa, beta))
            variable = kappa if var == 'kappa' else beta
            plot_limit_points(
                states[kappa, beta].products['limit points at 1.00'
                ].range(0.00, 4.00),
                title=None, label=fr'$\{var}={variable:.2f}$',
                finish=False, color=f'{c}')

        plt.legend(loc='center right')
        plt.xticks((0.0, 0.92, 1.8, 2.65, 3.6, 4.0))
        plt.yticks((0.0, 0.5, 1.0))
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("y")
        const_name = 'kappa' if var == 'beta' else 'beta'
        const_value = kappa if var == 'beta' else beta
        plt.title(fr'$\{const_name}={const_value:.2f}$')
        plt.title(subnumber, loc='left')

        show(output_path, name='var_' + var)


def case2():
    kappa = 1.
    beta = 100.
    # state = State.load(f"output/BOSK.pub/i2_kappa={kappa:.2f};beta={beta:.2f}")
    # plot_limit_points(
    #     state.products['limit points at 0.50'],
    #     title=fr'$\kappa={kappa}$ $\beta={beta}$', finish=False)
    state = State.load(f"output/BOSK.pub/ci2_kappa={kappa:.2f};beta={beta:.2f}")
    plot_limit_points(
        state.products['limit points at 0.50'],
        title=fr'$\kappa={kappa}$ $\beta={beta}$', finish=False)
    plt.grid()
    plt.show()

    kappa = 1.0

    states = []
    for beta in (0.0, 100.0):
        states.append(
            State.load(f"output/BOSK.pub/i2_kappa={kappa:.2f};beta={beta:.2f}"))

        plot_limit_points(
            states[-1].products['limit points at 0.50'],
            title=fr'$\kappa={kappa}$ $\beta={beta}$')
    #     intersect = states[-1].products['intersection at 0.50']
    #     results = tuple(intersect.data.items())
    #     T = results[-1][0]
    # num = len(results)
    # i = 0
    # for t, v in intersect.data.items():
    #     i += 1
    #     # if t in (s / 2 for s in range(int(2 * T) + 1)):
    #     plt.plot(*v, color=f'{1 - t / T:.3f}')
    #     if num * END[0] < i < num * END[1]:
    #         break
    # plt.title(f'{t:.2f}')
    # plt.show()


if __name__ == '__main__':
    # case1()
    case2()
