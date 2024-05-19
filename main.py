import numpy as np

import inputs
import grid_generation
import basis
import evolve 

# import initial_conditions
# import integrator
# import plots
# import test

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins,left_node_coordinates, right_node_coordinates)

# solving using continous galerkin method
cg_solution = evolve.CG_solver(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad)













coords_for_plot = np.unique(np.array(nodes_coordinates_phys_space).flatten())

coords_exact_solution = np.linspace(inputs.x_initial, inputs.x_final, 100)
exact_solution = ( 1 - np.exp( inputs.v * coords_exact_solution / inputs.a ) ) / ( 1 - np.exp( inputs.v / inputs.a ) )





import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import matplotlib as mpl
import matplotlib.ticker
from matplotlib.animation import FuncAnimation

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
ax.plot(coords_for_plot, cg_solution)
ax.plot(coords_exact_solution, exact_solution, linestyle = 'dashed')
fig.savefig('CG.pdf',bbox_inches='tight')
plt.clf()







'''



# generating initial conditions
h, u = initial_conditions.generate_initial_conditions(nodes_coordinates_phys_space)

# compute entropy : integral 0.5 * ( g * h**2 + h * u ) dx
entropy = evolve.compute_entropy(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, h, u)

# wrinting initial conditions file
integrator.write_data_file(element_number,nodes_coordinates_phys_space,entropy,h,u,False,0)

# compute mass matrix M_ij = integral phi_i(x) phi_j(x) dx and return the inverse matrix of M_ij
mass_matrix_inverse = evolve.compute_mass_matrix_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

# time step
time_step = np.array(inputs.t_limit/inputs.n_steps) 

# evolving in time the PDE
for number_of_t_step in np.arange(inputs.n_steps):

    # If true using euler method
    if inputs.evolution_method==0:
        h, u = integrator.euler_method( h, u, time_step, element_number, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, gauss_weights, element_lengths, basis_values_at_nodes, mass_matrix_inverse)

    # If true using RK4 method
    if inputs.evolution_method==1:
        h, u = integrator.rk4_method( h, u, time_step, element_number, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, gauss_weights, element_lengths, basis_values_at_nodes, mass_matrix_inverse)

    # saving the data
    if (number_of_t_step+1) % inputs.plot_every_steps == 0:

        # compute entropy : integral 0.5 * ( g * h**2 + h * u ) dx
        entropy = evolve.compute_entropy(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, h, u)

        # writing data
        integrator.write_data_file(element_number,nodes_coordinates_phys_space,entropy,h,u,False,number_of_t_step+1)

# plotting data
plots.plotting()

print(f'Done')

'''