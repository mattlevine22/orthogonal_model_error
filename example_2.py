import os
from rf import RF

LEG = ['1',
		'x',
		'(3*x**2 - 1) / 2',
		'(5*x**3 - 3*x) / 2',
		'(35*x**4 - 30*x**2 + 3) / 8',
		'(63*x**5 - 70*x**3 + 15*x) / 8']

## FROM PLUMLEE EMAIL
# % This one is simpler:
# y = @(x) 4*x+x.*sin(5*x); % target function
# f =@(x,theta) x*theta; %mechanistic model.
# theta = linspace(-5,10,100000);

fdag_str = '4*x + x * np.sin(5*x)' # target function
fig_path = 'outputs/example2'

lib_list = ['x'] # list of library functions for f0
orth_list = lib_list # list of functions for error term to be orthogonal to (typically, just the library list)

lam_lib = 0 # regularization (L1) on the library
lam_rf = 0 # regularization (L2) on the random feature function
zero_thresh = 1e-6 # inner-product threshold for discarding nearly-zero orthogonal features in Gram-Schmidt

# hyperparameters for variance of random feature function parameters w,b: tanh(wx + b)
Dr = 20 # number of random features
fac_w = 10
fac_b = 0.1

# coarsely sample data from the target function
data_step = 0.1
x_min = 0
x_max = 1

print('Target is 3.565236 x') # computed with standard OLS and data_step=0.0001


settings = {'lib_list': lib_list,
			'orth_list': orth_list,
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
			'x_max': x_max
			}

rf = RF(**settings)

rf.run()
