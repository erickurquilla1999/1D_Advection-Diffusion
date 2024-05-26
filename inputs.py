# Physical domain 
x_initial = 0 # (m) initial domain coordinate
x_final = 1 # (m) final domain coordinate

# DG method
N_elements = 4 # number of elements
p_basis_order = 2 # lagrange basis order

# Gauss cuadrature
n_gauss_poins = 5 

# Governing equation parameters
a = 1
Pe = 100
v = a * x_final / Pe

# Boundary conditions
u_at_x_initial = 0
u_at_x_final = 1