import numpy as np
import inputs
import basis

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

    s_mat_inv = np.linalg.inv(s_mat)
    x = s_mat_inv @ b

    return x

def DG_solver_advection(e_numb, e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature, nods_coords_phys_space):
    
    # evaluating the derivative in x of basis function evaluated in the gauss quadrature points
    # x_derivative_of_basis_func_at_gauss_quad_in_phys_space = [ [phi'_1(gauss_coords_1), phi'_2(gauss_coords_1) , ... , phi'_p(gauss_coords_1)], 
    #                                                               [phi'_1(gauss_coords_2), phi'_2(gauss_coords_2) , ... , phi'_p(gauss_coords_2)], ... , ]

    # evaluating the basis function in the gauss quadrature points
    # basis_func_values_at_gauss_quad_in_phys_space = [ [phi_1(gauss_coords_1), phi_2(gauss_coords_1) , ... , phi_p(gauss_coords_1)] , 
    #                                                   [phi_1(gauss_coords_2), phi_2(gauss_coords_2) , ... , phi_p(gauss_coords_2)] , ... , ]

    A_matrix = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    b_vector = np.zeros(   inputs.N_elements * ( inputs.p_basis_order + 1 )                                                      )

    ##################################################################
    # integral in advection term

    advection_matrix_1 = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for n in range(inputs.N_elements):
        for i in range( inputs.p_basis_order + 1 ):
            for j in range( inputs.p_basis_order + 1 ):
                advection_matrix_1[i + ( inputs.p_basis_order + 1 ) * n][j + ( inputs.p_basis_order + 1 ) * n] = inputs.a * 0.5 * e_lgth[n] * np.sum( g_weights[n] * ( bas_vals_x_der_at_gauss_quadrature[n][:,i] * bas_vals_at_gauss_quadrature[n][:,j] ) )
    
    ##################################################################
    # integral in diffusion term

    diffusion_matrix_1 = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for n in range(inputs.N_elements):
        for i in range( inputs.p_basis_order + 1 ):
            for j in range( inputs.p_basis_order + 1 ):
                diffusion_matrix_1[i + ( inputs.p_basis_order + 1 ) * n][j + ( inputs.p_basis_order + 1 ) * n] = inputs.v * 0.5 * e_lgth[n] * np.sum( g_weights[n] * ( bas_vals_x_der_at_gauss_quadrature[n][:,i] * bas_vals_x_der_at_gauss_quadrature[n][:,j] ) )

    ##################################################################
    # define boundary conditions

    state_leftboundary  = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) )
    state_rightboundary = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) )
    state_leftboundary [0 ] = inputs.u_at_x_initial
    state_rightboundary[-1] = inputs.u_at_x_final

    ##################################################################
    # upwind flux for advection

    uwf_matrix = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    uwf_vector = np.zeros(   inputs.N_elements * ( inputs.p_basis_order + 1 ) )

    # element at the left boundary
    uwf_leftterm_leftboundary  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    uwf_rightterm_leftboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    k=0
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            uwf_leftterm_leftboundary [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1]) * basis.lagrange_basis(nods_coords_phys_space[k], m, nods_coords_phys_space[k][-1])
            uwf_rightterm_leftboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0]) * basis.lagrange_basis(nods_coords_phys_space[k], m, nods_coords_phys_space[k][ 0])
    uwf_matrix += +1 * uwf_leftterm_leftboundary
    uwf_vector += -1 * uwf_rightterm_leftboundary @ state_leftboundary

    # interior elements
    uwf_leftterm_interiorelements  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    uwf_rightterm_interiorelements = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                uwf_leftterm_interiorelements [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1])
                uwf_rightterm_interiorelements[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0]) * basis.lagrange_basis(nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0])
    uwf_matrix = uwf_leftterm_interiorelements - uwf_rightterm_interiorelements
    
    # element at the right boundary
    uwf_leftterm_rightboundary  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    uwf_rightterm_rightboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            uwf_leftterm_rightboundary [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1])
            uwf_rightterm_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0]) * basis.lagrange_basis(nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0])
    uwf_matrix += -1 * uwf_rightterm_rightboundary
    uwf_vector += +1 * uwf_leftterm_rightboundary @ state_rightboundary

    ##################################################################
    # term 2 diffusion

    term2diff_matrix = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2diff_vector = np.zeros(   inputs.N_elements * ( inputs.p_basis_order + 1 ) )

    term1_leftterm_term2diff  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term1_rightterm_term2diff = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_leftterm_term2diff  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_rightterm_term2diff = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    term1_rightterm_term2diff_leftboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_rightterm_term2diff_leftboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    term1_leftterm_term2diff_rightboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_leftterm_term2diff_rightboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    # element at the left boundary
    k = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term1_rightterm_term2diff_leftboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v )     * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][0 ] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][0 ] )
            term2_rightterm_term2diff_leftboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v )     * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][0 ] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][0 ] )
            term1_leftterm_term2diff              [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term2_leftterm_term2diff              [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )

    # interior elements
    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                term1_leftterm_term2diff [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k    ], m, nods_coords_phys_space[k][-1] )
                term1_rightterm_term2diff[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k    ], m, nods_coords_phys_space[k][ 0] )
                term2_leftterm_term2diff [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k + 1], m, nods_coords_phys_space[k][-1] )
                term2_rightterm_term2diff[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k - 1], m, nods_coords_phys_space[k][ 0] )

    # element at the right boundary
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term1_leftterm_term2diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v )     * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k    ], m, nods_coords_phys_space[k][-1] )
            term2_leftterm_term2diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v )     * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k    ], m, nods_coords_phys_space[k][-1] )
            term1_rightterm_term2diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k    ], m, nods_coords_phys_space[k][ 0] )
            term2_rightterm_term2diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k - 1], m, nods_coords_phys_space[k][ 0] )
    
    term2diff_matrix = term1_leftterm_term2diff_rightboundary + term1_rightterm_term2diff_leftboundary + term1_leftterm_term2diff + term2_leftterm_term2diff
    term2diff_vector = term2_rightterm_term2diff_leftboundary @ state_leftboundary + term2_leftterm_term2diff_rightboundary @ state_rightboundary

    ##################################################################
    # term 3 diffusion
    
    term3diff_matrix = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3diff_vector = np.zeros(   inputs.N_elements * ( inputs.p_basis_order + 1 ) )

    #
    # term 1 of term 3 diffusion
    #

    term1_leftterm_term3diff               = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term1_rightterm_term3diff              = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term1_rightterm_term3diff_leftboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term1_leftterm_term3diff_rightboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    
    # element at the left boundary
    k = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term1_leftterm_term3diff              [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][-1] )
            term1_rightterm_term3diff_leftboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][ 0] )

    # interior elements
    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                term1_leftterm_term3diff          [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][-1] )
                term1_rightterm_term3diff         [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][ 0] )

    # element at the right boundary
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term1_leftterm_term3diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][-1] )
            term1_rightterm_term3diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k], m, nods_coords_phys_space[k][ 0] )

    term3diff_vector += term1_rightterm_term3diff_leftboundary @ state_leftboundary + term1_leftterm_term3diff_rightboundary @ state_rightboundary
    term3diff_matrix += term1_leftterm_term3diff + term1_rightterm_term3diff



    #
    # term 2 of term 3 diffusion
    #

    term2_leftterm_term3diff               = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_rightterm_term3diff              = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_rightterm_term3diff_leftboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term2_leftterm_term3diff_rightboundary = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    # element at the left boundary
    k = 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term2_leftterm_term3diff              [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
            term2_rightterm_term3diff_leftboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )

    # interior elements
    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                term2_leftterm_term3diff          [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
                term2_rightterm_term3diff         [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )

    # element at the right boundary
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term2_leftterm_term3diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term2_rightterm_term3diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )

    term3diff_vector += term2_rightterm_term3diff_leftboundary @ state_leftboundary + term2_leftterm_term3diff_rightboundary @ state_rightboundary
    term3diff_matrix += term2_leftterm_term3diff + term2_rightterm_term3diff









    print('\n\n\n\n')
    # so far
    A_matrix = uwf_matrix - advection_matrix_1 + diffusion_matrix_1 + term2diff_matrix + term3diff_matrix
    print(f'A_matrix = \n{A_matrix}')
    b_vector = uwf_vector + term2diff_vector + term3diff_vector
    print(f'b_vector = { -1 * b_vector}')

    s_mat_inv = np.linalg.inv(A_matrix)
    print(f's_mat_inv = {s_mat_inv}')

    x = -1 * s_mat_inv @ b_vector
    print(f'x = {x}')

    return 1