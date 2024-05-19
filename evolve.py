import numpy as np
import inputs

def CG_solver(e_numb, e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature):
    

    # evaluating the derivative in x of basis function evaluated in the gauss quadrature points
    # x_derivative_of_basis_func_at_gauss_quad_in_phys_space = [ [phi'_1(gauss_coords_1), phi'_2(gauss_coords_1) , ... , phi'_p(gauss_coords_1)], 
    #                                                               [phi'_1(gauss_coords_2), phi'_2(gauss_coords_2) , ... , phi'_p(gauss_coords_2)], ... , ]


    # evaluating the basis function in the gauss quadrature points
    # basis_func_values_at_gauss_quad_in_phys_space = [ [phi_1(gauss_coords_1), phi_2(gauss_coords_1) , ... , phi_p(gauss_coords_1)] , 
    #                                                   [phi_1(gauss_coords_2), phi_2(gauss_coords_2) , ... , phi_p(gauss_coords_2)] , ... , ]


    s_mat = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) ) )

    for n in range(inputs.N_elements):
        for i in range( inputs.p_basis_order + 1 ):
            for j in range( inputs.p_basis_order + 1 ):

                if ( i + inputs.p_basis_order * n == 0 and j == i ):
                    s_mat[i + inputs.p_basis_order * n][i + inputs.p_basis_order * n] = 1
                    break        
                elif ( i + inputs.p_basis_order * n == inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) - 1 ):
                    s_mat[i + inputs.p_basis_order * n][j + inputs.p_basis_order * n] = 0
                    if ( ( i + inputs.p_basis_order * n == inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) - 1 ) and j == i ):
                        s_mat[i + inputs.p_basis_order * n][j + inputs.p_basis_order * n] = 1      
                else:                      
                    s_mat[i + inputs.p_basis_order * n][j + inputs.p_basis_order * n] += 0.5 * e_lgth[n] * np.sum( g_weights[n] * ( - 1 * inputs.a * bas_vals_x_der_at_gauss_quadrature[n][:,i] * bas_vals_at_gauss_quadrature[n][:,j] + inputs.v * bas_vals_x_der_at_gauss_quadrature[n][:,i] * bas_vals_x_der_at_gauss_quadrature[n][:,j] ) )

    # setting boudary conditions
    b = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) )
    b[0] = inputs.u_at_x_initial
    b[inputs.N_elements * ( inputs.p_basis_order + 1 ) - ( inputs.N_elements - 1 ) - 1] = inputs.u_at_x_final

    x = np.linalg.solve(s_mat, b)

    return x














def compute_mass_matrix_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad):
    
    # in element k: M_ij = integral phi_i(x) phi_j(x) dx inside the element domain
    M = []
    M_inverse = []

    #Lopp over all element
    for n in elmnt_numb:
        phi = np.array(basis_values_at_gauss_quad[n])
        weights = gauss_weights[n]
        delta_x = element_lgth[n]
        
        # Compute M for the current element
        M_in_element_n = 0.5 * delta_x * np.dot(phi.T * weights, phi)
        # Append M to the list
        M.append(M_in_element_n)

        # Compute the inverse of M for the current element
        M_inv_in_element_n = np.linalg.inv(M_in_element_n)

        # Append the inverse of M to the list
        M_inverse.append(M_inv_in_element_n)
 
    return M_inverse

def compute_numerical_flux_vectors(element_n, basis_values_at_nods, h_, u_):

    # computing roe flux
    roe_flux_1 = []
    roe_flux_2 = []

    # looping over each element (except the final element) to compute the roe flux at the right
    for n in element_n[:-1]:

        # computing average value in the right border of element n and n+1
        h_average = 0.5*(h_[n][-1]+h_[n+1][0])
        u_average = 0.5*(u_[n][-1]+u_[n+1][0])

        # compute the jacobian evaluated in the border between elements n and n+1
        jacobian = [ [ 0 , 1 ] , [ inputs.g * h_average - u_average**2, 2 * u_average ] ]

        # compute eigenvalues and eigenvector of the jacobian
        eigenvalues_jacobian, eigenvectors_jacobian = np.linalg.eig(jacobian)

        # biulds abs_A matrix
        abs_A = eigenvectors_jacobian @ np.diag(np.abs(eigenvalues_jacobian)) @ np.linalg.inv(eigenvectors_jacobian)

        # for the border between element n and n+1 compute the f1 on the left and the right
        f1_left  = h_[n][-1] * u_[n][-1] 
        f1_right = h_[n+1][0] * u_[n+1][0]

        # for the border between element n and n+1 compute the f2 on the left and the right
        f2_left  = h_[n][-1] * u_[n][-1]**2 + 0.5 * inputs.g * h_[n][-1]**2
        f2_right = h_[n+1][0] * u_[n+1][0]**2 + 0.5 * inputs.g * h_[n+1][0]**2

        # for the border between element n and n+1 compute the u1 on the left and the right
        u1_left  = h_[n][-1]
        u1_right = h_[n+1][0]

        # for the border between element n and n+1 compute the u2 on the left and the right
        u2_left  = h_[n][-1] * u_[n][-1] 
        u2_right = h_[n+1][0] * u_[n+1][0]

        # compute roe flux
        roe_flux_1.append( 0.5 * ( f1_left + f1_right ) - 0.5 * abs_A[0][0] * ( u1_right - u1_left ) - 0.5 * abs_A[0][1] * ( u2_right - u2_left ) )
        roe_flux_2.append( 0.5 * ( f2_left + f2_right ) - 0.5 * abs_A[1][0] * ( u1_right - u1_left ) - 0.5 * abs_A[1][1] * ( u2_right - u2_left ) )

    # computing the difference between the numerical fluxe in limits of the element
    difference_numerical_flux_1 = []
    difference_numerical_flux_2 = []

    #looping over all element
    for n in element_n:
        # compute differences between flux: right numerical flux - left numerical flux
        if n == 0:               
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * roe_flux_1[n] - basis_values_at_nods[n][:,0] * 0 )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * roe_flux_2[n] - basis_values_at_nods[n][:,0] * ( 0.5 * inputs.g * h_[n][0]**2 ) )
        elif n == element_n[-1]: 
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * 0 - basis_values_at_nods[n][:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * ( 0.5 * inputs.g * h_[n][-1]**2 ) - basis_values_at_nods[n][:,0] * roe_flux_2[n-1] )
        else:                    
            difference_numerical_flux_1.append( basis_values_at_nods[n][:,-1] * roe_flux_1[n] - basis_values_at_nods[n][:,0] * roe_flux_1[n-1] )
            difference_numerical_flux_2.append( basis_values_at_nods[n][:,-1] * roe_flux_2[n] - basis_values_at_nods[n][:,0] * roe_flux_2[n-1] )

    return np.array(difference_numerical_flux_1), np.array(difference_numerical_flux_2)

def compute_stiffness_vectors(e_numb,e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature, _h, _u):
    
    # number of basis or nodes in each element
    number_of_basis = len(bas_vals_at_gauss_quadrature[0][0])

    # interpolate h from nodes to quadrature points
    _h_at_gau_quad = [ bas_at_gau_quad @ __h for bas_at_gau_quad, __h in zip(bas_vals_at_gauss_quadrature, _h)]

    # interpolate u from nodes to quadrature points
    _u_at_gau_quad = [ bas_at_gau_quad @ __u for bas_at_gau_quad, __u in zip(bas_vals_at_gauss_quadrature, _u)]

    # compute stiffness vectors
    stiff_vec_1 = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_x_der_at_gauss_quadrature[n][:,i] * ( _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])                                                   ) ) for i in range(number_of_basis) ] for n in e_numb]
    stiff_vec_2 = [ [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * bas_vals_x_der_at_gauss_quadrature[n][:,i] * ( _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])**2 + 0.5 * inputs.g * np.array(_h_at_gau_quad[n])**2 ) ) for i in range(number_of_basis) ] for n in e_numb]
    
    return stiff_vec_1, stiff_vec_2

def compute_time_derivatives(h__, u__, ele_nub, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, gau_wei, ele_len, bas_vals_at_nod, mass_matrix_inverse__):
    
    # computing stiffness vectors
    stiffness_vector_1_, stiffness_vector_2_ = compute_stiffness_vectors(ele_nub, ele_len, gau_wei, bas_vals_at_gau_quad, bas_vals_x_der_at_gau_quad, h__, u__)

    # computing numerical flux
    numerical_flux_vector_1_, numerical_flux_vector_2_ = compute_numerical_flux_vectors(ele_nub, bas_vals_at_nod, h__, u__)

    # computing residual vector
    residual_vector_1_ = stiffness_vector_1_ - numerical_flux_vector_1_
    residual_vector_2_ = stiffness_vector_2_ - numerical_flux_vector_2_

    # compute time derivatives of h and hu
    dh_dt_ = [mass_mat_inv__ @ res_vec_1_ for mass_mat_inv__, res_vec_1_ in zip(mass_matrix_inverse__, residual_vector_1_)]
    dhu_dt_ = [mass_mat_inv__ @ res_vec_2_ for mass_mat_inv__, res_vec_2_ in zip(mass_matrix_inverse__, residual_vector_2_)]

    # compute time derivatives of u
    du_dt_ = np.where( h__ == 0 , 0 , ( dhu_dt_ - u__ * dh_dt_ ) / h__ )

    return dh_dt_, du_dt_

def compute_entropy(e_numb,e_lgth, g_weights, bas_vals_at_gauss_quadrature, _h, _u):

    # interpolate h from nodes to quadrature points
    _h_at_gau_quad = [ bas_at_gau_quad @ __h for bas_at_gau_quad, __h in zip(bas_vals_at_gauss_quadrature, _h)]

    # interpolate u from nodes to quadrature points
    _u_at_gau_quad = [ bas_at_gau_quad @ __u for bas_at_gau_quad, __u in zip(bas_vals_at_gauss_quadrature, _u)]

    # compute entropy
    entropy_ = np.sum ( [ 0.5 * e_lgth[n] * np.sum( g_weights[n] * 0.5 * ( inputs.g * np.array(_h_at_gau_quad[n])**2 + _h_at_gau_quad[n] * np.array(_u_at_gau_quad[n])**2 ) ) for n in e_numb] )

    return entropy_