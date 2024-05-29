import numpy as np

import inputs
import grid_generation
import basis
import evolve 
import plots

# creating mesh
element_number, left_node_coordinates, right_node_coordinates, nodes_coordinates_phys_space, nodes_coordinates_ref_space, element_lengths = grid_generation.generate_1d_mesh(inputs.x_initial,inputs.x_final,inputs.N_elements,inputs.p_basis_order)

# generating reference space information
gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, basis_values_at_nodes, gauss_coords_phys_space = basis.generate_reference_space(element_number,nodes_coordinates_phys_space,inputs.n_gauss_poins,left_node_coordinates, right_node_coordinates)

# solving using continous galerkin method
cg_solution = evolve.CG_solver(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad)

# compute mass matrix 1 : M_ij = integral phi_i(x) phi_j(x) dx and return the inverse
mass_matrix_inverse = evolve.compute_mass_matrix_inverse(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad)

# solving using discontinous galerkin method
dg_solution = evolve.DG_solver(element_number, element_lengths, gauss_weights, basis_values_at_gauss_quad, basis_values_x_derivative_at_gauss_quad, nodes_coordinates_phys_space, mass_matrix_inverse)

# plot solution
plots.plot(cg_solution, dg_solution, nodes_coordinates_phys_space, basis_values_at_gauss_quad, gauss_coords_phys_space, gauss_weights)