import utilities
import inputs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import matplotlib as mpl
import matplotlib.ticker
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2

def plot(cg_solution, dg_solution, nodes_coordinates_phys_space):

    coords_for_plot = np.unique(np.array(nodes_coordinates_phys_space).flatten())
    coords_exact_solution = np.linspace(inputs.x_initial, inputs.x_final, 100)
    exact_solution = ( 1 - np.exp( inputs.a * coords_exact_solution / inputs.v ) ) / ( 1 - np.exp( inputs.a / inputs.v ) )
    dg_cords = nodes_coordinates_phys_space
    dg_solut = dg_solution.reshape(inputs.N_elements, inputs.p_basis_order + 1)

    fig, ax = plt.subplots()
    ax.plot(coords_for_plot, cg_solution, label = 'CG', color = 'C0')
    for i in range(inputs.N_elements):
        label = 'DG' if i == 0 else None
        ax.plot(dg_cords[i], dg_solut[i], color='C1', label=label)
    ax.plot(coords_exact_solution, exact_solution, linestyle = 'dotted', label = 'Exact', color = 'black')
    leg = ax.legend(framealpha=0.0,ncol=1)
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')
    fig.savefig('solution.pdf',bbox_inches='tight')
    plt.clf()