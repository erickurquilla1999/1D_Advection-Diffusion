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

def plot(cg_solution, dg_solution, nodes_coordinates_phys_space, basis_values_at_gauss_quad, gauss_coords_phys_space, gauss_weights):

    ###############################################
    # Compute error
    dg_cords = nodes_coordinates_phys_space
    dg_solut = dg_solution.reshape(inputs.N_elements, inputs.p_basis_order + 1)

    cg_solut = []
    for i in range(inputs.N_elements):
        cg_solut.append(list(cg_solution[ i * inputs.p_basis_order : i * inputs.p_basis_order + inputs.p_basis_order + 1]))
        
    cg_quadrature = []
    dg_quadrature = []
    for i in range( inputs.N_elements ):
        cg_quadrature.append( basis_values_at_gauss_quad[i] @ cg_solut[i] )
        dg_quadrature.append( basis_values_at_gauss_quad[i] @ dg_solut[i] )

    w = ( 1 - np.cos( np.pi * gauss_coords_phys_space / inputs.x_final) ) / 2
    es_quadrature = ( 1 - np.exp( inputs.a * gauss_coords_phys_space / inputs.v ) ) / ( 1 - np.exp( inputs.a / inputs.v ) )

    cg_weighted_error = 0
    dg_weighted_error = 0
    cg_l2_error = 0
    dg_l2_error = 0

    for i in range( inputs.N_elements ):
        # weighted error
        cg_weighted_error += ( 0.5 / inputs.N_elements ) * np.sum( gauss_weights[i] * w[i] * ( cg_quadrature[i] - es_quadrature[i] ) )
        dg_weighted_error += ( 0.5 / inputs.N_elements ) * np.sum( gauss_weights[i] * w[i] * ( dg_quadrature[i] - es_quadrature[i] ) )
        # l2 error
        cg_l2_error += ( 0.5 / inputs.N_elements ) * np.sum( gauss_weights[i] * ( cg_quadrature[i] - es_quadrature[i] ) ** 2 )
        dg_l2_error += ( 0.5 / inputs.N_elements ) * np.sum( gauss_weights[i] * ( dg_quadrature[i] - es_quadrature[i] ) ** 2 )

    cg_weighted_error = np.abs (cg_weighted_error)
    dg_weighted_error = np.abs (dg_weighted_error)
    cg_l2_error       = np.sqrt(cg_l2_error)
    dg_l2_error       = np.sqrt(dg_l2_error)

    ###############################################
    # Plot solution
    
    fig, ax = plt.subplots()

    # cg solution
    cg_cords = np.unique(np.array(nodes_coordinates_phys_space).flatten())
    ax.plot(cg_cords, cg_solution, label = 'CG', color = 'C0')

    # dg solution
    for i in range(inputs.N_elements):
        label = 'DG' if i == 0 else None
        ax.plot(dg_cords[i], dg_solut[i], color='C1', label=label)

    # exact solution
    coords_exact_solution = np.linspace(inputs.x_initial, inputs.x_final, 100)
    exact_solution = ( 1 - np.exp( inputs.a * coords_exact_solution / inputs.v ) ) / ( 1 - np.exp( inputs.a / inputs.v ) )
    ax.plot(coords_exact_solution, exact_solution, linestyle = 'dotted', label = 'Exact', color = 'black')

    # plot setting
    leg = ax.legend(framealpha=0.0,ncol=1)
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_linewidth(0.0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')
    fig.savefig('solution.pdf',bbox_inches='tight')
    plt.clf()

    ###############################################
    # Print information

    dg_cords = list(dg_cords)
    dg_solut = list(dg_solut)
    for i in range(inputs.N_elements):
        dg_cords[i] = list(dg_cords[i])
        dg_solut[i] = list(dg_solut[i])

    print(f'#Ne={inputs.N_elements}')
    print(f'#Pe={inputs.Pe}')
    print(f'#p={inputs.p_basis_order}')

    print(f'cg_l2_error_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {cg_l2_error}')
    print(f'cg_weighted_error_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {cg_weighted_error}')

    print(f'dg_l2_error_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {dg_l2_error}')
    print(f'dg_weighted_error_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {dg_weighted_error}')
    
    print(f'cg_cords_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {list(cg_cords)}')
    print(f'cg_solution_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {list(cg_solution)}')
    
    print(f'dg_cords_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {dg_cords}')
    print(f'dg_solut_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {dg_solut}')

    print(f'coords_exact_solution_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {list(coords_exact_solution)}')
    print(f'exact_solution_{inputs.N_elements}_{inputs.p_basis_order}_{inputs.Pe} = {list(exact_solution)}')