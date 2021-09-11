import os
import numpy as np
import cvxpy as cp
from scipy.linalg import pinv2 as scipypinv2
from scipy.integrate import quad, nquad
# Plotting parameters
import matplotlib
from matplotlib import rc, cm, colors
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
import pickle
import warnings

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
					fdag,
					lib_list,
					lib_list_str = ['1', 'x', 'x**2'],
					fdag_str = ' + '.join(LEG[:5]),
					integration_ranges=None,
					x_input = None,
					y = None,
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

		orth_list = lib_list
		orth_list_str = lib_list_str
		np.random.seed(seed)

		self.fig_path = fig_path
		os.makedirs(self.fig_path, exist_ok=True)

		if orth_list is None:
			orth_list = lib_list

		self.orth_list = orth_list

		self.lib_list = lib_list
		self.lib_list_str = lib_list_str
		# self.lib_len = len(self.lib_list)

		self.fdag_str = fdag_str
		self.fdag = fdag
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
		self.Dl = len(self.orth_list)
		self.zero_thresh = zero_thresh
		self.do_normalization = do_normalization

		if integration_ranges is None:
			integration_ranges = [(self.x_min_grid,self.x_max_grid) for _ in range(self.Dx)]
		self.integration_ranges = integration_ranges

		self.make_mesh()

		if x_input is None or y is None:
			self.make_data()
		else:
			self.x = x_input
			self.y = y

		self.Nx = self.x.shape[1]
		self.lambda_reg_lib = lam_lib * self.Nx / (self.Dy)
		self.lambda_reg_rf = lam_rf * self.Nx / (self.Dy)

	def run(self):
		# self.make_data()
		self.normalize_data()
		self.fit_data()
		self.plot_fits()

		self.print_eval()

	def make_data(self):
		if self.Dx==2:
			self.x = np.mgrid[self.x_min:self.x_max:self.data_step, self.x_min:self.x_max:self.data_step].reshape(2,-1)
		elif self.Dx==3:
			self.x = np.mgrid[self.x_min:self.x_max:self.data_step, self.x_min:self.x_max:self.data_step, self.x_min:self.x_max:self.data_step].reshape(3,-1)
		else:
			if self.data_grid is None:
				self.x = np.arange(start=self.x_min, stop=self.x_max, step=self.data_step) # input
			else:
				self.x = self.data_grid
			self.x = self.x[None,:]
		self.y = self.fdag(self.x.T).T # output

	def make_mesh(self):
		if self.Dx==1:
			self.x_grid = np.arange(start=self.x_min_grid, stop=self.x_max_grid, step=self.grid_step) # input
			self.x_grid = self.x_grid[:,None]
		elif self.Dx==2:
			x = np.arange(start=self.x_min_grid, stop=self.x_max_grid, step=self.grid_step)
			y = np.arange(start=self.x_min_grid, stop=self.x_max_grid, step=self.grid_step)
			self.X_grid, self.Y_grid = X, Y = np.meshgrid(x,y)
			self.XY_grid = np.array([X.flatten(),Y.flatten()]).T
		else:
			print('Cant make mesh for larger than 2-d')

	def normalize_data(self):
		self.x_norm = self.scaleX(self.x, save=True)
		self.y_norm = self.scaleY(self.y, save=True)

	def fit_data(self, use_cvx=False):
		self.set_rf()

		### Compute libraries
		print('Computing Phi and Library matrices...')
		Phi = self.Phi_lib(self.x_norm.T).T
		self.plot_rf_seq(Phi.T[:500], nm='rf_seq')

		Phi_lib = self.F_lib(self.x_norm.T).T
		self.plot_rf_seq(Phi_lib.T[:500], nm='lib_seq')

		self.set_rf_orth()
		print('Computing Phi-orth matrix...')
		Phi_orth = self.Phi_orth_lib(self.x_norm.T).T
		self.plot_rf_seq(Phi_orth.T[:500], nm='rf_orth_seq')

		### Library regression
		print('Running regressions...')
		if use_cvx:
			# L1
			cv = CVOPT()
			C_lib = cv.run_lasso(X=Phi_lib, Y=self.y_norm, lambd=self.lambda_reg_lib)
		else:
			# L2 penalty
			Ireg = self.lambda_reg_lib * np.eye(self.Dl)
			C_lib = self.y_norm @ Phi_lib.T @ scipypinv2(Phi_lib@Phi_lib.T + Ireg)
		#aggregate
		y_pred_scaled = C_lib @ Phi_lib
		self.y_fit_lib_only = self.descaleY(C_lib @ Phi_lib)
		# create regression fit string
		foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list_str[k]) for k in range(self.Dl)]
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

		# ### Fit whole function with RF-ORTHOGONAL (just for testing stuff)
		# if use_cvx:
		# 	cv = CVOPT()
		# 	C = cv.run_ridge(X=Phi_orth, Y=self.y_norm, lambd=self.lambda_reg_rf)
		# else:
		# 	C = self.y_norm @ Phi_orth.T @ scipypinv2(Phi_orth@Phi_orth.T + self.lambda_reg_rf * np.eye(self.Dr))
		# #aggregate
		# y_pred_scaled = C @ Phi_orth
		self.y_fit_rf_orth_only = self.descaleY(y_pred_scaled)


		### joint: L1-Lib + L2-OrthRF learning
		if Phi_orth.shape[0]>0:
			if use_cvx:
				cv = CVOPT()
				C_lib, C_orth = cv.run_mixed(X1=Phi_lib, X2=Phi_orth, Y=self.y_norm, lambd_1=self.lambda_reg_lib, lambd_2=self.lambda_reg_rf)
			else:
				warnings.warn("regularization not supported here for joint learning without CVX")
				Phi_all = np.vstack((Phi_lib, Phi_orth))
				C_all = self.y_norm @ Phi_all.T @ scipypinv2(Phi_all@Phi_all.T + self.lambda_reg_rf * np.eye(self.Dr + self.Dl))
				C_lib = C_all[:,:self.Dl]
				C_orth = C_all[:,self.Dl:]

			y_pred_scaled = C_lib @ Phi_lib + C_orth @ Phi_orth
			self.y_fit_joint_orth = self.descaleY(y_pred_scaled)
			# create regression fit string
			foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list_str[k]) for k in range(self.Dl)]
			self.regression_string_joint_orth = ' + '.join(foo)
		else:
			self.y_fit_joint_orth = np.zeros_like(y_pred_scaled)
			self.regression_string_joint_orth = 'failed'

		### joint: L1-Lib + L2-RF learning (non-orth)
		if use_cvx:
			cv = CVOPT()
			C_lib, C_orth = cv.run_mixed(X1=Phi_lib, X2=Phi, Y=self.y_norm, lambd_1=self.lambda_reg_lib, lambd_2=self.lambda_reg_rf)
		else:
			warnings.warn("regularization not supported here for joint learning without CVX")
			Phi_all = np.vstack((Phi_lib, Phi))
			C_all = self.y_norm @ Phi_all.T @ scipypinv2(Phi_all@Phi_all.T + self.lambda_reg_rf * np.eye(self.Dr + self.Dl))
			C_lib = C_all[:,:self.Dl]
			C_orth = C_all[:,self.Dl:]
		y_pred_scaled = C_lib @ Phi_lib + C_orth @ Phi
		self.y_fit_joint = self.descaleY(y_pred_scaled)
		# create regression fit string
		foo = ['{:.3e} {}'.format(C_lib[0,k], self.lib_list_str[k]) for k in range(self.Dl)]
		self.regression_string_joint = ' + '.join(foo)

	def f_j(self, j, x_input):
		return self.lib_list[j](x_input)

	def f_orth_j(self, j, x_input):
		return self.lib_orth_list[j](x_input)

	def phi_j(self, j, x_input):
		# x_input: N x Dx
		# phi_J: N x Dx -> N x 1
		N = x_input.shape[0]

		# build Wj: Dx x 1
		Wj = self.w_in[j].reshape(self.Dx,1)

		# build bj: N x 1
		bj = self.b_in[j] * np.ones((N,1))

		# compute random feature
		foo = np.cos(x_input @ Wj + bj)
		return foo

	def F_lib(self, x_input):
		# x_input: N x Dx
		# output:  N x Dl
		N = x_input.shape[0]
		foo = np.zeros((N, self.Dl))
		for l in range(self.Dl):
			foo[:,l] = np.squeeze(self.lib_list[l](x_input))
		return foo

	def F_orth_lib(self, x_input):
		# x_input: N x Dx
		# output:  N x Dl
		N = x_input.shape[0]
		foo = np.zeros((N, self.Dl))
		for l in range(self.Dl):
			foo[:,l] = np.squeeze(self.lib_orth_list[l](x_input))
		return foo

	def Phi_lib(self, x_input):
		# x_input: N x Dx
		# output:  N x Dr
		N = x_input.shape[0]
		foo = np.zeros((N, self.Dr))
		for j in range(self.Dr):
			foo[:,j] = np.squeeze(self.phi_j(j, x_input))
		return foo

	def Phi_orth_lib(self, x_input):
		# x_input: N x Dx
		# output:  N x Dr
		N = x_input.shape[0]
		foo = np.zeros((N, self.Dr))
		for j in range(self.Dr):
			foo[:,j] = np.squeeze(self.phi_orth_j(j, x_input))
		return foo

	def phi_orth_j(self, j, x_input):
		# x_input: N x Dx
		# output: N x 1
		proj_j = self.F_orth_lib(x_input) @ self.alpha_proj[j].reshape(self.Dl,1)
		foo = self.phi_j(j, x_input) - proj_j
		foo = foo / self.phi_orth_norm[j]
		return foo

	def get_alpha_proj(self):
		# produces alpha_proj: Dr x Dl
		# phi_orth_norm: Dr

		self.alpha_proj = np.zeros((self.Dr, self.Dl))
		self.phi_orth_norm = np.ones(self.Dr)

		print('Computing alphas')
		for l in tqdm(range(self.Dl)):
			fl = self.lib_orth_list[l]
			fl_norm = self.norm(fl) # this should be ~1 already, maybe remove this?
			for j in range(self.Dr):
				alp = self.ip(fl, lambda x: self.phi_j(j, x))
				self.alpha_proj[j,l] = alp / fl_norm**2

		# next, we normalize the phi_orth_j
		print('Normalizing Phi_orth...')
		for j in tqdm(range(self.Dr)):
			self.phi_orth_norm[j] = self.norm(lambda x: self.phi_orth_j(j, x))


	def set_rf(self):
		self.w_in = np.random.uniform(low=-self.fac_w, high=self.fac_w, size= (self.Dr, self.Dx))
		self.b_in = np.random.uniform(low=-self.fac_b, high=self.fac_b, size= (self.Dr, 1))

		self.f_phi = lambda x: np.cos(self.w_in @ x + self.b_in)
		self.f_phi_list = [self.f_phi_listmaker(i) for i in range(self.Dr)]
		if self.Dx==1:
			self.plot_rf_d1(self.x_grid, self.Phi_lib(self.x_grid), nm='rf_functions')
		elif self.Dx==2:
			self.plot_rf_d2(f=self.phi_j, J=self.Dr, nm='rf_functions')
			self.plot_rf_d2(f=self.f_j, J=self.Dl, nm='lib_functions')
		else:
			print('Couldnt plot RF')

	def f_phi_listmaker(self, i):
		return lambda x: np.asarray(np.cos(self.w_in[i] @ x + self.b_in[i])).reshape(-1)

	def f_0_listmaker(self, f0_str):
		return lambda x: np.asarray(eval(f0_str)).reshape(-1)

	def set_rf_orth(self, ollie=True):
		print('Running GS on library...')
		self.lib_orth_list = self.gram_schmidt(self.lib_list)

		print('Getting projections of RFs wrt library...')
		self.get_alpha_proj()
		if self.Dx==1:
			self.plot_rf_d1(self.x_grid, self.Phi_orth_lib(self.x_grid), nm='rf_orth_functions')
		elif self.Dx==2:
			self.plot_rf_d2(f=self.f_orth_j, J=self.Dl, nm='lib_orth_functions')
			self.plot_rf_d2(f=self.phi_orth_j, J=self.Dr, nm='rf_orth_functions')
		else:
			print('Couldnt plot RF')

	def ip(self, u, v):
		'''compute euclidean innerproduct of u, v on interval (self.x_min_grid, self.x_max_grid)'''
		# foo_int, err_int = quad(func = lambda x: u(x)*v(x), a=self.x_min_grid, b=self.x_max_grid, limit=5000)

		def myFunc(*argv):
			# input is x0, x1, x2...
			# output is a float
			x = np.asarray(argv).reshape(1,self.Dx)
			foo = u(x)*v(x)
			return np.float(foo)

		foo_int, err_int = nquad(func = myFunc, ranges=self.integration_ranges, opts={'limit':500})
		return foo_int

	def norm(self, u):
		return np.sqrt(self.ip(u, u))

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
				f_orth_list_norm += [1] # now they are all orthonormal
				k_eff += 1

		return f_orth_list

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
		plt.close()

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

	def plot_rf_d1(self, x,  Phi, nm='RF_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		N, D = Phi.shape
		for d in range(D):
			ax.plot(x, Phi[:,d], label='f_{}'.format(d))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))
		plt.close()

	def plot_rf_seq(self, Phi, nm='RF_seq_functions'):
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		N, D = Phi.shape
		for d in range(min(10,D)):
			ax.plot(Phi[:,d], label='f_{}'.format(d))
			plt.savefig(os.path.join(self.fig_path, nm))
		ax.legend()
		plt.savefig(os.path.join(self.fig_path, nm))
		plt.close()


	def plot_rf_d2(self, f, J, nm='RF_functions'):
		for j in range(min(J,10)):
			Phi = f(j, self.XY_grid)
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
			ax.scatter(x=self.XY_grid[:,0], y=self.XY_grid[:,1], c=Phi)
			plt.savefig(os.path.join(self.fig_path, nm+str(j)))
			plt.close()

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
