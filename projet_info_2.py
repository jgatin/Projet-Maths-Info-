#Pas fixe

import numpy as np
import matplotlib.pyplot as plt

def solve_euler_explicit(f, dt, tf, x0, t0 = 0): 
	t = np.arange(t0, tf, dt)
	x = [x0]
	for temps in t[:-1]:
		x.append(x[-1] + dt * f(temps, x[-1]))
	return t, np.array(x)


def ecart_max(u, f, dt, tf, x0, t0 = 0):
	t, x = u(f, dt, tf, x0, t0 = 0)
	e = []
	for i, temp in enumerate(t):
		e.append(abs(x[i] - np.sin(temp)))
	return max(e)

def f(t, x):
	return np.cos(x)

pas = np.arange(0.001, 0.1, 0.001)

#on remarquera que au bout d'un moment c'est plus linéaire 

e = []
for dt in pas:
	e.append(ecart_max(solve_euler_explicit, f, 1, dt, 10))


plt.plot(pas, e, '.', color = 'pink')
plt.show()

def solve_heun_explicit(f, dt, tf, x0, t0 = 0):
	t = np.arange(t0, tf, dt)
	x = [x0]
	for i, temp in enumerate(t[:-1]):
		x.append(x[-1] + dt/2 * (f(temp, x[-1]) + f(t[i+1], x[-1] + dt * f(temp, x[-1]))))
	return t, np.array(x)


T = 10

t, x = solve_euler_explicit(f, 0.001, T, 0, t0 = 0)
u, v = solve_heun_explicit(f, 0.001, T, 0, t0 = 0)
k = [np.sin(temp) for temp in t]
plt.plot(t, x, color = 'blue')
plt.plot(t, v, color = 'red')
plt.plot(t, k, color = 'green')
plt.show()


pas = np.arange(0.001, 0.1, 0.001)

e = []
for dt in pas:
	e.append(ecart_max(solve_heun_explicit, f, 1, dt, 10))


plt.plot(pas, e, '.', color = 'blue')
plt.show()


