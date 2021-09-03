import os
import numpy as np
import cvxpy as cp
from scipy.linalg import pinv2 as scipypinv2
from scipy.integrate import quad
# Plotting parameters
import matplotlib
from matplotlib import rc, cm, colors
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
import pickle

from sklearn.linear_model import LinearRegression

import warnings
from tqdm import tqdm

try:
	from pyrfm import OrthogonalRandomFeature, StructuredOrthogonalRandomFeature
except:
	pass


from pdb import set_trace as bp

LEG = ['1',
		'x',
		'(3*x**2 - 1) / 2',
		'(5*x**3 - 3*x) / 2',
		'(35*x**4 - 30*x**2 + 3) / 8',
		'(63*x**5 - 70*x**3 + 15*x) / 8'] #legendre polynomials (orthogonal)


class RF(object):
	def __init__(self,
					lib_list = ['1', 'x', 'x**2'],
					orth_list = LEG[:2],
					fdag_str = ' + '.join(LEG[:5]),
					Dx = 1,
					Dr = 20,
					fac_w = 2,
					fac_b = 2,
					x_min = -1,
					x_max = 1,
					data_grid = None,
					data_step = 0.01,
					grid_step = 1e-2,
					lam_lib = 1e-5,
					lam_rf = 1e-5,
					zero_thresh = 1e-7,
					do_normalization = False,
					fig_path= 'test4_output',
					seed=0):
		'''lib_list: mechanistic dictionary library
			orth_list: list of functions for RF method to orthogonalize against.
						Default is orth_list = lib_list
			fdag_str: determines the full true target function

		'''
		np.random.seed(seed)

		self.fig_path = fig_path
		os.makedirs(self.fig_path, exist_ok=True)

		if orth_list is None:
			orth_list = lib_list

		self.orth_list = orth_list
		self.orth_len = len(self.orth_list)

		self.lib_list = lib_list
		self.lib_len = len(self.lib_list)

		self.fdag_str = fdag_str
		self.fac_w = fac_w
		self.fac_b = fac_b
		self.x_min = x_min
		self.x_max = x_max
		self.x_min_grid = 1*x_min
		self.x_max_grid = 1*x_max
		self.data_step = data_step
		self.data_grid = data_grid
		self.grid_step = grid_step
		self.Dx = Dx
		self.Dy = 1
		self.Dr = Dr
		self.zero_thresh = zero_thresh
		self.do_normalization = do_normalization

		self.integration_ranges = [(self.x_min_grid,self.x_max_grid) for _ in range(self.Dx)]

		if self.Dx==1:
			self.make_mesh()

		self.make_data()

		self.Nx = self.x.shape[1]
		self.lambda_reg_lib = lam_lib * self.Nx / (self.Dy)
		self.lambda_reg_rf = lam_rf * self.Nx / (self.Dy)

	def run(self):
		self.make_data()
		self.normalize_data()
		self.fit_data()
		self.plot_fits()

		self.print_eval()

	def make_data(self):
		if self.Dx==2:
			self.x = np.mgrid[self.x_min:self.x_max:self.data_step, self.x_min:self.x_max:self.data_step].reshape(2,-1)
		else:
			if self.data_grid is None:
				self.x = np.arange(start=self.x_min, stop=self.x_max, step=self.data_step) # input
			else:
				self.x = self.data_grid
			self.x = self.x[None,:]
		self.y = self.fdag(self.x) # output

	def make_mesh(self):
		self.x_grid = np.arange(start=self.x_min_grid, stop=self.x_max_grid, step=self.grid_step) # input

	def normalize_data(self):
		self.x_norm = self.scaleX(self.x, save=True)
		self.y_norm = self.scaleY(self.y, save=True)

	def fit_data(self, use_cvx=True):
		self.set_rf()

		### Compute libraries
		Phi = self.f_phi(self.x_norm) # compute Phi
		# Phi = self.Phi_sorf
		self.set_rf_orth()
		Phi_lib = self.get_library(x=self.x_norm, my_list=self.f_lib_list) # compute value of polynomial regression library
		Phi_orth = self.get_library(x=self.x_norm, my_list=self.f_phi_orth_list) # compute value of random feature function library

		### Library regression
		if use_cvx:
			# L1
			cv = CVOPT()
			C_lib = cv.run_lasso(X=Phi_lib, Y=self.y_norm, lambd=self.lambda_reg_lib)
		else:
			# L2 penalty
			Ireg = self.lambda_reg_lib * np.eye(self.lib_len)
			C_lib = self.y_norm @ Phi_lib.T @ scipypinv2(Phi_lib@Phi_lib.T + Ireg)
		#aggregate
		y_pred_scaled = C_lib @ Phi_lib
		self.y_fit_lib_only = self.descaleY(C_lib @ Phi_lib)
		# create regression fit string
		foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list[k]) for k in range(self.lib_len)]
		self.regression_string_lib_only = ' + '.join(foo)


		### Fit whole function with RF
		if use_cvx:
			cv = CVOPT()
			C = cv.run_ridge(X=Phi, Y=self.y_norm, lambd=self.lambda_reg_rf)
		else:
			C = self.y_norm @ Phi.T @ scipypinv2(Phi@Phi.T + self.lambda_reg_rf * np.eye(self.Dr))
		#aggregate
		y_pred_scaled = C @ Phi
		self.y_fit_rf_only = self.descaleY(y_pred_scaled)

		### Fit whole function with RF-ORTHOGONAL (just for testing stuff)
		if use_cvx:
			cv = CVOPT()
			C = cv.run_ridge(X=Phi_orth, Y=self.y_norm, lambd=self.lambda_reg_rf)
		else:
			C = self.y_norm @ Phi_orth.T @ scipypinv2(Phi_orth@Phi_orth.T + self.lambda_reg_rf * np.eye(self.Dr))
		#aggregate
		y_pred_scaled = C @ Phi_orth
		self.y_fit_rf_orth_only = self.descaleY(y_pred_scaled)


		### joint: L1-Lib + L2-OrthRF learning
		if Phi_orth.shape[0]>0:
			cv = CVOPT()
			C_lib, C_orth = cv.run_mixed(X1=Phi_lib, X2=Phi_orth, Y=self.y_norm, lambd_1=self.lambda_reg_lib, lambd_2=self.lambda_reg_rf)
			y_pred_scaled = C_lib @ Phi_lib + C_orth @ Phi_orth
			self.y_fit_joint_orth = self.descaleY(y_pred_scaled)
			# create regression fit string
			foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list[k]) for k in range(self.lib_len)]
			self.regression_string_joint_orth = ' + '.join(foo)
		else:
			self.y_fit_joint_orth = np.zeros_like(y_pred_scaled)
			self.regression_string_joint_orth = 'failed'

		### joint: L1-Lib + L2-RF learning (non-orth)
		cv = CVOPT()
		C_lib, C_orth = cv.run_mixed(X1=Phi_lib, X2=Phi, Y=self.y_norm, lambd_1=self.lambda_reg_lib, lambd_2=self.lambda_reg_rf)
		y_pred_scaled = C_lib @ Phi_lib + C_orth @ Phi
		self.y_fit_joint = self.descaleY(y_pred_scaled)
		# create regression fit string
		foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list[k]) for k in range(self.lib_len)]
		self.regression_string_joint = ' + '.join(foo)


	def get_library(self, x, my_list):
		J = len(my_list)
		if J==0:
			warnings.warn('Library is of length 0.')
		Phi = np.zeros((J, self.Nx))
		for j in range(J):
			fj = my_list[j]
			Phi[j] = fj(x)
		return Phi

	def set_rf(self):
		self.w_in = np.random.uniform(low=-self.fac_w, high=self.fac_w, size= (self.Dr, self.Dx))
		self.b_in = np.random.uniform(low=-self.fac_b, high=self.fac_b, size= (self.Dr, 1))

		self.f_phi = lambda x: np.cos(self.w_in @ x + self.b_in)
		self.f_phi_list = [self.f_phi_listmaker(i) for i in range(self.Dr)]
		try:
			self.plot_rf(self.f_phi_list, nm='rf_functions')
		except:
			print('Couldnt plot RF')

		# get SORF stuff (not using yet)
		# self.transformer_sorf = StructuredOrthogonalRandomFeature(n_components=self.Dr, gamma=10,
		# 						distribution='gaussian',
		# 						random_state=0, use_offset=True)
		# self.Phi_sorf = self.transformer_sorf.fit_transform(self.x_norm.T).T
		# self.plot_sorf(nm='sorf')

	def f_phi_listmaker(self, i):
		return lambda x: np.cos(self.w_in[i] * x + self.b_in[i])

	def f_0_listmaker(self, f0_str):
		return lambda x: eval(f0_str)

	def set_rf_orth(self, ollie=True):
		# set library functions
		self.f_lib_list = [self.f_0_listmaker(f0_str) for f0_str in self.lib_list]
		self.f_orth_list = [self.f_0_listmaker(f0_str) for f0_str in self.orth_list]

		if ollie:
			self.f_phi_orth_list = self.gram_schmidt_ollie(orth_list=self.f_orth_list, rf_list=self.f_phi_list)
		else:
			f_list_gs = self.f_orth_list + self.f_phi_list
			f_orth_list = self.gram_schmidt(f_list_gs)
			self.f_phi_orth_list = f_orth_list[self.orth_len:] # don't include f0 in the list

		self.Dr_orth = len(self.f_phi_orth_list)
		self.plot_rf(self.f_phi_orth_list, nm='rf_orth_functions')


	def f0(self, x):
		return eval(self.f0_str)

	def fdag(self, x):
		return eval(self.fdag_str)

	def ip(self, u, v):
		'''compute euclidean innerproduct of u, v on interval (self.x_min_grid, self.x_max_grid)'''
		foo_int, err_int = quad(func = lambda x: u(x)*v(x), a=self.x_min_grid, b=self.x_max_grid, limit=5000)
		return foo_int

	def compute_alpha(self, u, v):
		# project function v orthogonally onto the span of function u
		return alpha

	def f_update(self, f, alpha, u):
		'''update function by subtracting projection'''
		f_upd = lambda x: f(x) - alpha*u(x)
		return f_upd

	def f_normalize(self, f, nrm):
		'''normalize function by dividing by its norm'''
		f_nrm = lambda x: f(x) / nrm
		return f_nrm

	def gram_schmidt(self, f_list):
		'''run gram-schmidt orthogonalization on list of functions.
		see wikipedia for pseudo-code: https://en.wikipedia.org/wiki/Gram–Schmidt_process.'''
		f_orth_list = []
		f_orth_list_norm = []
		k_eff = 0 # keep track of number of non-zero orthogonal functions
		for k in tqdm(range(len(f_list))):
			f_orth = f_list[k]
			for i in tqdm(range(k_eff), leave=False): # skipped for k=0
				ui = f_orth_list[i]
				unorm = f_orth_list_norm[i]
				alpha = self.ip(ui, f_list[k]) / unorm
				f_orth = self.f_update(f=f_orth, alpha=alpha, u=ui)

			f_orth_sq_norm = self.ip(f_orth, f_orth)
			f_orth = self.f_normalize(f=f_orth, nrm=np.sqrt(f_orth_sq_norm))

			# print('|f_{k}|^2 ='.format(k=k), f_orth_sq_norm)
			if f_orth_sq_norm > self.zero_thresh: # only keep functions that arent close to zero.
				f_orth_list += [f_orth]
				# f_orth_list_norm += [f_orth_sq_norm]
				f_orth_list_norm += [1] # now they are all orthonormal
				k_eff += 1
			# else:
				# print('|f_{k}|^2 ='.format(k=k), f_orth_sq_norm)
				# print('did not meet zero_threshold!')
		return f_orth_list

	def gram_schmidt_ollie(self, orth_list, rf_list):
		'''run gram-schmidt orthogonalization FIRST on library functions.
		THEN, on each random feature individually wrt the orthogonal library.
		see wikipedia for pseudo-code: https://en.wikipedia.org/wiki/Gram–Schmidt_process.'''

		# first get orthogonal versions of orth_list
		orth_list = self.gram_schmidt(orth_list)

		# now, loop over phi's
		rf_list_orth = []

		for rf in tqdm(rf_list):
			for ui in orth_list:
				unorm = 1
				# unorm = self.ip(ui,ui) # it is already 1 from previous run of gram_scmidt above!!!
				alpha = self.ip(ui, rf) / unorm
				rf = self.f_update(f=rf, alpha=alpha, u=ui)

			rf_sq_norm = self.ip(rf, rf)
			rf = self.f_normalize(f=rf, nrm=np.sqrt(rf_sq_norm))

			if rf_sq_norm > self.zero_thresh: # only keep functions that arent close to zero.
				rf_list_orth += [rf]

		return rf_list_orth

	def scaleX(self, x, save=False):
		if self.do_normalization:
			if save:
				self.x_mean = np.mean(x)
				self.x_std = np.std(x)
			return (x-self.x_mean) / self.x_std
		else:
			return x

	def descaleX(self, x):
		if self.do_normalization:
			return self.x_mean + (self.x_std * x)
		else:
			return x

	def scaleY(self, y, save=False):
		if self.do_normalization:
			if save:
				self.y_mean = np.mean(y)
				self.y_std = np.std(y)
			return (y-self.y_mean) / self.y_std
		else:
			return y

	def descaleY(self, y):
		if self.do_normalization:
			return self.y_mean + (self.y_std * y)
		else:
			return y

	def plot_fits(self):
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,6))

		ax[0].plot(self.x.T, self.y.T, linestyle='-', linewidth=4, label='True: {}'.format(self.fdag_str))
		ax[0].plot(self.x.T, self.y_fit_rf_only.T, linestyle='--', linewidth=4, label='General RF regression')
		ax[0].plot(self.x.T, self.y_fit_rf_orth_only.T, linestyle='--', linewidth=4, label='Orthongal RF regression ONLY')
		ax[0].plot(self.x.T, self.y_fit_joint.T, linestyle='--', linewidth=4, label='Joint regression: Library + RF')
		ax[0].plot(self.x.T, self.y_fit_joint_orth.T, linestyle='--', linewidth=4, label='Joint-Orth regression: Library + RF-Orth')
		ax[0].plot(self.x.T, self.y_fit_lib_only.T, linestyle=':', color='black', linewidth=4, label='Library-only regression')
		ax[0].set_ylabel('f(x)')
		ax[0].set_xlabel('x')
		ax[0].legend()

		# ax[1].plot(self.x.T, self.y_fit_joint_orth.T, linestyle='-', linewidth=4, label='RF component: orthogonal to {}'.format(self.orth_list))
		# ax[1].plot(self.x.T, self.y_fit_joint_lib.T, linestyle='-', linewidth=4, label='Library component = ' + self.regression_string_joint)
		ax[1].plot(self.x.T, self.y_fit_lib_only.T, linestyle=':', color='black', linewidth=4, label='Library only = ' + self.regression_string_lib_only)
		ax[1].set_title('Fits: Dr={Dr}'.format(Dr=self.Dr))
		ax[1].set_ylabel('f(x)')
		ax[1].set_xlabel('x')
		ax[1].legend()
		plt.savefig(os.path.join(self.fig_path, 'fits'))

	def print_eval(self):
		print("True function:", self.fdag_str)
		print("Regular Library Regression:", self.regression_string_lib_only)
		print("Joint (non-orth) Library Regression:", self.regression_string_joint)
		print("Joint-Orth Library Regression :", self.regression_string_joint_orth)
		print('\n')

		print("RF-pure MSE = {:.1e}".format(np.mean((self.y - self.y_fit_rf_only)**2)))
		print("Joint Lib + RF MSE = {:.1e}".format(np.mean((self.y - self.y_fit_joint)**2)),)
		print("Joint Lib + RF-Orth MSE= {:.1e}".format(np.mean((self.y - self.y_fit_joint_orth)**2)),)
		print("Lib-Only MSE = {:.3e}".format(np.mean((self.y - self.y_fit_lib_only)**2)))

	def plot_rf(self, func_list, nm='RF_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		c = -1
		for f in func_list:
			c += 1
			ax.plot(self.x_grid, f(self.x_grid), label='f_{}'.format(c))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))

	def plot_sorf(self, nm='SORF_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		array = self.transformer_sorf.fit_transform(self.x_grid[:,None]).T
		for c in range(array.shape[0]):
			ax.plot(self.x_grid, array[c,:], label='f_{}'.format(c))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))

def nquad_wrapper(func, ranges, args=None, opts=None, full_output=False):

	def myFunc(*argv):
		x = np.asarray(argv)
		foo = func(x)
		return np.asarray(foo)

	result, abs_err = nquad(myFunc, ranges=ranges, args=args, opts=opts, full_output=full_output)
	return result, abs_err


class CVOPT(object):
	def __init__(self):
		return

	def loss_mixed(self, X1, X2, Y, A1, A2):
		return cp.pnorm(A1@X1 + A2@X2 - Y, p=2)**2

	def loss_single(self, X, Y, A):
		return cp.pnorm(Y - A @ X, p=2)**2

	def l1_reg(self, A1):
		return cp.norm1(A1)

	def l2_reg(self, A2):
		return cp.pnorm(A2, p=2)**2

	def objective_mixed(self, X1, X2, Y, A1, A2, lambd_1, lambd_2):
		return self.loss_mixed(X1, X2, Y, A1, A2) + lambd_1*self.l1_reg(A1) + lambd_2*self.l2_reg(A2)

	def objective_ridge(self, X2, Y, A2, lambd_2):
		return self.loss_single(X2, Y, A2) + lambd_2*self.l2_reg(A2)

	def objective_lasso(self, X1, Y, A1, lambd_1):
		return self.loss_single(X1, Y, A1) + lambd_1*self.l1_reg(A1)

	def mse_mixed(self, X1, X2, Y, A1, A2):
		return (1.0 / Y.shape[1]) * self.loss_mixed(X1, X2, Y, A1, A2).value

	def mse_single(self, X, Y, A):
		return (1.0 / Y.shape[1]) * self.loss_single(X, Y, A).value

	def run_mixed(self, X1, X2, Y, lambd_1, lambd_2):
		A1 = cp.Variable((Y.shape[0], X1.shape[0]))
		A2 = cp.Variable((Y.shape[0], X2.shape[0]))
		lam1 = cp.Parameter(nonneg=True)
		lam2 = cp.Parameter(nonneg=True)
		problem = cp.Problem(cp.Minimize(self.objective_mixed(X1, X2, Y, A1, A2, lam1, lam2)))
		lam1.value = lambd_1
		lam2.value = lambd_2
		problem.solve()
		return A1.value, A2.value

	def run_ridge(self, X, Y, lambd):
		A2 = cp.Variable((Y.shape[0], X.shape[0]))
		lam = cp.Parameter(nonneg=True)
		problem = cp.Problem(cp.Minimize(self.objective_ridge(X, Y, A2, lam)))
		lam.value = lambd
		problem.solve()
		return A2.value

	def run_lasso(self, X, Y, lambd):
		A1 = cp.Variable((Y.shape[0], X.shape[0]))
		lam = cp.Parameter(nonneg=True)
		problem = cp.Problem(cp.Minimize(self.objective_lasso(X, Y, A1, lam)))
		lam.value = lambd
		problem.solve()
		return A1.value

	def run_sk_reg(self, X, Y):
		reg = LinearRegression().fit(X, Y)
		foo = np.stack((reg.coef_,reg.intercept_))
		return foo

	def plot_train_test_errors(train_errors, test_errors, lambd_values):
		plt.plot(lambd_values, train_errors, label="Train error")
		plt.plot(lambd_values, test_errors, label="Test error")
		plt.xscale("log")
		plt.legend(loc="upper left")
		plt.xlabel(r"$\lambda$", fontsize=16)
		plt.title("Mean Squared Error (MSE)")
		plt.show()

	def plot_regularization_path(lambd_values, beta_values):
		num_coeffs = len(beta_values[0])
		for i in range(num_coeffs):
			plt.plot(lambd_values, [wi[i] for wi in beta_values])
		plt.xlabel(r"$\lambda$", fontsize=16)
		plt.xscale("log")
		plt.title("Regularization Path")
		plt.show()



if __name__ == '__main__':
	rf = RF()
	rf.run()
