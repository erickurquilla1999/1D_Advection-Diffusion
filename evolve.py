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
    # upwind flux for advection
    
    x_cords = np.array(nods_coords_phys_space).flatten()

    up_wind_flux   = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )

    # element at the left boundary
    up_wind_flux_l_br = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    up_wind_flux_r_br = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for k in [e_numb[0]]:
        print(f'k = {k}')
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                up_wind_flux_l_br[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order])
                up_wind_flux_r_br[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * k + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k                       ]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, x_cords[(inputs.p_basis_order + 1) * k                       ])
    up_wind_flux += up_wind_flux_l_br

    # interior elements
    up_wind_flux_l = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    up_wind_flux_r = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for k in e_numb[1:-1]:
        print(f'k = {k}')
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                up_wind_flux_l[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order])
                up_wind_flux_r[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k                       ]) * basis.lagrange_basis(nods_coords_phys_space[k-1], m, x_cords[(inputs.p_basis_order + 1) * k                       ])
    up_wind_flux = up_wind_flux_l - up_wind_flux_r
    
    # element at the right boundary
    up_wind_flux_l_bl = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    up_wind_flux_r_bl = np.zeros( ( inputs.N_elements * ( inputs.p_basis_order + 1 ) , inputs.N_elements * ( inputs.p_basis_order + 1 ) ) )
    for k in [e_numb[-1]]:
        print(f'k = {k}')
        for n in range(inputs.p_basis_order + 1):
            for m in range(inputs.p_basis_order + 1):
                up_wind_flux_l_bl[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) *   k       + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order]) * basis.lagrange_basis(nods_coords_phys_space[k  ], m, x_cords[(inputs.p_basis_order + 1) * k + inputs.p_basis_order])
                up_wind_flux_r_bl[ ( inputs.p_basis_order + 1 ) * k + n ][ ( inputs.p_basis_order + 1 ) * ( k - 1 ) + m ] = inputs.a * basis.lagrange_basis(nods_coords_phys_space[k], n, x_cords[(inputs.p_basis_order + 1) * k                       ]) * basis.lagrange_basis(nods_coords_phys_space[k-1], m, x_cords[(inputs.p_basis_order + 1) * k                       ])
    up_wind_flux -= up_wind_flux_r_bl

    # build vector b
    b_upwind = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) )
    state_bl = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) )
    state_br = np.zeros( inputs.N_elements * ( inputs.p_basis_order + 1 ) )
    state_br[ 0 ]                 = inputs.u_at_x_initial
    state_bl[ len(state_br) - 1 ] = inputs.u_at_x_final
    b_upwind = - up_wind_flux_l_bl @ state_bl + up_wind_flux_r_br @ state_br

    ##################################################################
    # 
    
    return 1