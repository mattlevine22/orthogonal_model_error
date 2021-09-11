import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.optimize as optimization

def f(state, t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def func(params, xdata, ydata):
    return (ydata - np.dot(xdata, params))

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 400.0, 0.01)
states = odeint(f, state0, t)

x = states[:,0]
y = states[:,1]
z = states[:,2]

## Setting the output data
#dy_dt = x * (rho) - y
dy_dt = x * (rho - z) - y
ydata = dy_dt

## Setting the input data
xdata = np.vstack((x,y)).T
#xdata = np.vstack((x,y,z)).T
#xdata = np.vstack((x,y,x*z)).T
#xdata = np.vstack((x,y,x*z + np.random.normal(0,1e1,x.shape))).T

## Least square fit
theta0 = np.zeros(xdata.shape[1])
print(optimization.leastsq(func, theta0, args=(xdata, ydata)))
