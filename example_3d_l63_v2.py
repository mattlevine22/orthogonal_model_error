import os
import numpy as np
from rf_new import RF
from pdb import set_trace as bp
from odelibrary import L63

fig_path = 'outputs/example_3d_l63_v2'

##### This example is designed to use trajectory data to estimate 28*x, -1*y under missing "-xz" term.



Dx = 3 # input dimension

# true function
def fdag(x, rho=28):
	# x_input: N x Dx
	# output: N x 1 (constant)
	# rho x - y - xz
	foo = rho * x[:,0] - x[:, 1] - x[:,0]*x[:,2]
	foo = foo.reshape(-1,1)
	return foo
fdag_str = '28*x - y - xz' # target function

# constant library term
def I_i(x_input):
	# x_input: N x Dx
	# output: N x 1 (constant)
	N = x_input.shape[0]
	foo = np.ones((N,1))
	return foo

# linear library term
def X_i(x_input, i):
	# x_input: N x Dx
	# output: N x 1
	N = x_input.shape[0]
	foo = x_input[:,i].reshape(N,1)
	return foo

# build list of library functions
lib_list = [lambda x: X_i(x, 0), lambda x: X_i(x, 1)]

# build list of library function names as strings for printing
lib_list_str = []
lib_list_str += ['x_{}'.format(i) for i in range(2)]

# lib_list = ['x[{}]'.format(c) for c in range(Dx)] # list of library functions for f0
orth_list = lib_list # list of functions for error term to be orthogonal to (typically, just the library list)

lam_lib = 0 # regularization (L1) on the library
lam_rf = 0 # regularization (L2) on the random feature function
zero_thresh = 0 # inner-product threshold for discarding nearly-zero orthogonal features in Gram-Schmidt

# hyperparameters for variance of random feature function parameters w,b: tanh(wx + b)
Dr = 20 # number of random features
fac_w = 0.1
fac_b = 2*np.pi

# coarsely sample data from the target function
# data_step = 0.1
# x_min = -10
# x_max = 10

delta_t = 0.01
t_data = 100 # length of training trajectory in model time units
t_transient = 100 # burn-in length for data generation
rng_seed = 0 # =None for random
ode = L63()
X, Xdot = ode.generate_data(delta_t=delta_t, t_transient=t_transient, t_data=t_data, rng_seed=rng_seed)
integration_ranges = [(np.min(X[:,i]), np.max(X[:,i])) for i in range(Dx)]

# set output data
y = Xdot[:,1,None] # output data

# rescale inputs
print('Target is 28 x1 - 1 x2') # computed with standard OLS and data_step=0.0001
sd_X = np.std(X,axis=0)
X = X / sd_X
print('Target (/sd) is {} x1 - {} x2'.format(28*sd_X[0], sd_X[1]))



settings = {'lib_list': lib_list,
			'lib_list_str': lib_list_str,
			'integration_ranges': integration_ranges,
			'x_input': X.T,
			'y': y.T,
			# 'orth_list': orth_list,
			# 'orth_list_str': lib_list_str,
			'fdag': fdag,
			'fdag_str': fdag_str,
			'fig_path': fig_path,
			'Dr': Dr,
			'lam_lib': lam_lib,
			'lam_rf': lam_rf,
			'zero_thresh': zero_thresh,
			'fac_w': fac_w,
			'fac_b': fac_b,
			# 'data_step': data_step,
			# 'x_min': x_min,
			# 'x_max': x_max,
			'Dx': Dx
			}

rf = RF(**settings)

rf.run()
