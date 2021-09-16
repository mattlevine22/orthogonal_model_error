import os
import numpy as np
from rf import RF
from pdb import set_trace as bp

fig_path = 'outputs/example_4x_xsin5x_1d'

Dx = 1 # input dimension

# true function
def fdag(x):
	# x_input: N x Dx
	# output: N x 1 (constant)
	foo = 4*x + x*np.sin(5*x)
	foo = foo.reshape(-1,1)
	return foo
fdag_str = '4x + x sin(5x)' # target function


def I_i(x_input):
	# x_input: N x Dx
	# output: N x 1 (constant)
	N = x_input.shape[0]
	foo = np.ones((N,1))
	return foo

def X_i(x_input, i):
	# x_input: N x Dx
	# output: N x 1
	N = x_input.shape[0]
	foo = x_input[:,i].reshape(N,1)
	return foo

lib_list = []
lib_list += [lambda x: X_i(x, i) for i in range(Dx)]

lib_list_str = []
lib_list_str += ['x_{}'.format(i) for i in range(Dx)]


# lib_list = ['x[{}]'.format(c) for c in range(Dx)] # list of library functions for f0
orth_list = lib_list # list of functions for error term to be orthogonal to (typically, just the library list)

lam_lib = 0 # regularization (L1) on the library
lam_rf = 0 # regularization (L2) on the random feature function
zero_thresh = 0 # inner-product threshold for discarding nearly-zero orthogonal features in Gram-Schmidt

# hyperparameters for variance of random feature function parameters w,b: tanh(wx + b)
Dr = 20 # number of random features
fac_w = 1
fac_b = 2*np.pi

# coarsely sample data from the target function
data_step = 0.1
x_min = 0
x_max = 1

print('Target is 3.565236 x') # computed with standard OLS and data_step=0.0001


settings = {'lib_list': lib_list,
			'lib_list_str': lib_list_str,
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
			'data_step': data_step,
			'x_min': x_min,
			'x_max': x_max,
			'Dx': Dx
			}

rf = RF(**settings)

rf.run()
