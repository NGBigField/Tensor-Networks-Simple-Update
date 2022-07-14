from simple_update import SimpleUpdate
import matplotlib.pyplot as plt
from os.path import join
import numpy as np


def plot_convergence_curve(simple_update: SimpleUpdate, figname='su_simulation_plot', figure_size=(17, 10),
                           floating_point_error=7):
    """
    Plot the convergence curve and energy of a simple update experiment
    :param simple_update:
    :param figname:
    :param figure_size:
    :param floating_point_error:
    :return:
    """
    plt.rcParams.update({'font.size': 18})

    error = simple_update.logger['error']
    energy = simple_update.logger['energy']
    dt = simple_update.logger['dt']
    iteration = simple_update.logger['iteration']

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(211, label='1')
    ax2 = fig.add_subplot(211, label='2', frame_on=False)
    ax3 = fig.add_subplot(212)

    if simple_update.tensor_network.network_name is not None:
        ax.set_title(simple_update.tensor_network.network_name)
    ax.plot(iteration, error, color='C0')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Convergence error', color='C0')
    ax.tick_params(axis='x', color='C0')
    ax.tick_params(axis='y', color='C0')

    ax2.plot(iteration, energy, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Energy (per-site)', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', color='C1')
    argmin_energy = np.argmin(energy)
    ax2.plot(iteration[argmin_energy], energy[argmin_energy], 'o', color='red',
             label=r'$E_0 = $' + str(np.round(energy[argmin_energy], floating_point_error)))
    ax2.legend()

    ax3.plot(iteration, dt, '.')
    ax3.set_yscale('log')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('dt (ITE interval)')

    if simple_update.tensor_network.network_name is not None:
        plt.savefig(join(simple_update.tensor_network.dir_path, simple_update.tensor_network.network_name) + '.png')
    else:
        plt.savefig(join(simple_update.tensor_network.dir_path, figname) + '.png')

    plt.show()


def l2(x: np.array, y: np.array):
    """
    Computes the Euclidean distance (L2 norm) between x and y.
    :param x: np.array
    :param y: np.array
    :return: Euclidean distance
    """
    return np.sqrt(np.sum(np.square(x - y)))
