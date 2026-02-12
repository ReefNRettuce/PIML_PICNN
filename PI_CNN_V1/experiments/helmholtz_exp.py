# This is the helmholtz loss function and output 

# first I need to define a patch 

# then I define a local function 

# the local function is the g vector times the A vector which is a matrix of coefficients 

# define the g vector 

# Defining the A vector 
# to define the A vector we need the pressure field, the x direction slope which is the partial derivative with respe
# to x 

# the du_y which is the partial derivative with respect to y 

# du_2_x_y I'm pretty sure this is the laplacian 

#this defines the a matrices 

#then I need to also find teh wavenumber 

# then I need to find the Kronecker matrices 

#c1 is a 4*4 matrix 
#coefficients are 
#  C_1 = [ (1,1/2,1/3,1/4), (1/2,1/3,1/4,1/5), (1/3,1/4, 1/5, 1/6) , (1/4,1/5,1/6, 1/7)
#  (l, l^2, l^3, l^4), (l^2,L^3,L^4,L^5), (L^3, L^4, L^5, L^6), (L^4, L^5, L^6, L^7)

#C2 is a 4*4 matrix 
# C_2 = [(0,0,0,0), (0,0,0,0), (0,0,4*L^2, 6*L^2), (0,0,6*L^2, 12*L^3)]

#C3 is a 4*4 matrix 
# C_3 = [(0,0,0,0), (0,0,0,0), (2*L^2, L^2, 2*L^3 * 1/3, L^4 * 1/2), (3*L^2, 2*L^3, 3/2 * L^4, 6/5 * L^5)]

#then the loss function Lh is derived from 
# A*C_1*(A)^T kron C_2
# A*C_2*(A)^T kron C_1 
# K^4 * A * C_1 * (A)^T kron C_3
# 2 * A * (C_3)^T * (A)^T kron C_3
# 2 * K^2 * C_3 * (A)^T kron C_1 
# all of that times so lambda 

