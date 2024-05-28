import numpy as np
import inputs
import basis

def compute_mass_matrix_1_inverse(elmnt_numb,element_lgth, gauss_weights, basis_values_at_gauss_quad):

    # number of basis or nodes in each element
    number_of_basis = len(basis_values_at_gauss_quad[0][0])

    # compute mass matrix 1
    M1 = [ [ [ 0.5 * element_lgth[n] * np.sum( basis_values_at_gauss_quad[n][:,i] * gauss_weights[n] * basis_values_at_gauss_quad[n][:,j] ) for j in range(number_of_basis) ] for i in range(number_of_basis) ] for n in elmnt_numb]

    # compute inverse of mass matrix 2
    M1_inverse = [ np.linalg.inv(M1[n]) for n in elmnt_numb]
    
    return M1_inverse


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

def DG_solver_advection(e_numb, e_lgth, g_weights, bas_vals_at_gauss_quadrature, bas_vals_x_der_at_gauss_quadrature, nods_coords_phys_space, mass_matr_inv):

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
                term1_leftterm_term3diff          [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
                term1_rightterm_term3diff         [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )

    # element at the right boundary
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term1_leftterm_term3diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term1_rightterm_term3diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )

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
                term2_rightterm_term3diff         [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )

    # element at the right boundary
    k = e_numb[-1]
    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term2_leftterm_term3diff_rightboundary[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( -1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term2_rightterm_term3diff             [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( +1 * inputs.v / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis_derivative( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )

    term3diff_vector += term2_rightterm_term3diff_leftboundary @ state_leftboundary + term2_leftterm_term3diff_rightboundary @ state_rightboundary
    term3diff_matrix += term2_leftterm_term3diff + term2_rightterm_term3diff

    #
    # term 3 of term 3 diffusion
    #

    mass_matrix_inverse = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for k in e_numb:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                mass_matrix_inverse[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] = mass_matr_inv[k][n][m]

    '''
    delta_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    Phi_1_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    Phi_2_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    Phi_3_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    Phi_4_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    
    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                Phi_1_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
                Phi_2_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
                Phi_3_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
                Phi_4_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )

    print(f'Phi_1_plus = \n{Phi_1_plus}')
    print(f'Phi_2_plus = \n{Phi_2_plus}')
    print(f'Phi_3_plus = \n{Phi_3_plus}')
    print(f'Phi_4_plus = \n{Phi_4_plus}')

    gamma_1_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    gamma_2_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                for q in range(inputs.p_basis_order + 1):
                    gamma_1_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += mass_matrix_inverse[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + q ] * Phi_1_plus[ ( inputs.p_basis_order + 1 ) * k + q ][ ( inputs.p_basis_order + 1 ) *   k       + m ]
                    gamma_2_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += mass_matrix_inverse[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + q ] * Phi_2_plus[ ( inputs.p_basis_order + 1 ) * k + q ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ]

    print(f'gamma_1_plus = \n{gamma_1_plus}')
    print(f'gamma_2_plus = \n{gamma_2_plus}')
    
    delta_plus = 0.5 * gamma_1_plus - 0.5 * gamma_2_plus

    print(f'delta_plus = \n{delta_plus}')
    '''





    term3diff_matrix = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3diff_vector = np.zeros(   inputs.N_elements * ( inputs.p_basis_order + 1 ) )

    # element at the left boundary

    delta_plus    = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    delta_minus   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    delta_minus_b = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    k = 0

    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            delta_plus   [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += (   1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            delta_plus   [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
            delta_minus_b[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( + 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
            delta_minus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )

    delta_plus    = mass_matrix_inverse @ delta_plus
    delta_minus   = mass_matrix_inverse @ delta_minus
    delta_minus_b = mass_matrix_inverse @ delta_minus_b

    term3_term3diff_plus   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_plus_b = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_minu   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term3_term3diff_plus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term3_term3diff_plus_b[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
            term3_term3diff_minu  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
            term3_term3diff_minu  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
    
    term3diff_matrix += term3_term3diff_plus @ delta_plus + term3_term3diff_minu @ delta_minus
    term3diff_vector += term3_term3diff_plus_b @ delta_minus_b @ state_leftboundary

    # interior elements

    delta_plus  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    delta_minus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                delta_plus [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += (   1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
                delta_plus [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
                delta_minus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += (   1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )
                delta_minus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
    
    delta_plus  = mass_matrix_inverse @ delta_plus
    delta_minus = mass_matrix_inverse @ delta_minus

    print(f'delta_plus  = \n{delta_plus}')
    print(f'delta_minus = \n{delta_minus}')

    term3_term3diff = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_plus = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_minu = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    for k in e_numb[1:-1]:
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                term3_term3diff_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
                term3_term3diff_plus[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )
                term3_term3diff_minu[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k + 1 ) + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k+1], m, nods_coords_phys_space[k][-1] )
                term3_term3diff_minu[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
    
    term3diff_matrix += term3_term3diff_plus @ delta_plus + term3_term3diff_minu @ delta_minus

    # element at the right boundary

    delta_plus   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    delta_minus  = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    delta_plus_b = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    k = e_numb[-1]

    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            delta_plus   [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += (   1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            delta_plus_b [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            delta_minus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += (   1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )
            delta_minus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += ( - 1 / 2 ) * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
    
    delta_plus    = mass_matrix_inverse @ delta_plus
    delta_minus   = mass_matrix_inverse @ delta_minus
    delta_plus_b = mass_matrix_inverse @ delta_plus_b

    term3_term3diff_plus   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_minu_b = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    term3_term3diff_minu   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    for n in range(inputs.p_basis_order + 1):
        for m in range(inputs.p_basis_order + 1):
            term3_term3diff_plus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term3_term3diff_plus  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k-1], m, nods_coords_phys_space[k][ 0] )
            term3_term3diff_minu_b[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += +inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][-1] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][-1] )
            term3_term3diff_minu  [ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] += -inputs.v * basis.lagrange_basis( nods_coords_phys_space[k], n, nods_coords_phys_space[k][ 0] ) * basis.lagrange_basis( nods_coords_phys_space[k  ], m, nods_coords_phys_space[k][ 0] )
    
    term3diff_matrix += term3_term3diff_plus @ delta_plus + term3_term3diff_minu @ delta_minus
    term3diff_vector += term3_term3diff_minu_b @ delta_plus_b @ state_rightboundary

    ##################################################################
    # solve

    A_matrix = uwf_matrix - advection_matrix_1 + diffusion_matrix_1 + term2diff_matrix + term3diff_matrix + term3_term3diff
    b_vector = uwf_vector + term2diff_vector + term3diff_vector
    s_mat_inv = np.linalg.inv(A_matrix)
    x = -1 * s_mat_inv @ b_vector

    return x