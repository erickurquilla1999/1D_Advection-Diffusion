import numpy as np

import inputs
import grid_generation
import basis
import evolve 

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins,left_node_coordinates, right_node_coordinates)

# solving using continous galerkin method
cg_solution = evolve.CG_solver(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad)

# compute mass matrix 1 : M_ij = integral phi_i(x) phi_j(x) dx and return the inverse
mass_matrix_1_inverse = evolve.compute_mass_matrix_1_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

# solving using discontinous galerkin method
dg_solution = evolve.DG_solver_advection(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, nodes_coordinates_phys_space, mass_matrix_1_inverse)

################################################################
# plotting

coords_for_plot = np.unique(np.array(nodes_coordinates_phys_space).flatten())
coords_exact_solution = np.linspace(inputs.x_initial, inputs.x_final, 100)
exact_solution = ( 1 - np.exp( inputs.a * coords_exact_solution / inputs.v ) ) / ( 1 - np.exp( inputs.a / inputs.v ) )
dg_cords = nodes_coordinates_phys_space
dg_solut = dg_solution.reshape(inputs.N_elements, inputs.p_basis_order + 1)

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