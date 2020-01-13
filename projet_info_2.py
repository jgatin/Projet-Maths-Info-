#Pas fixe

import numpy as np
import matplotlib.pyplot as plt

def solve_euler_explicit(f, dt, tf, x0, t0 = 0): 
    t, x = [t0], [x0]
    while t[-1] < tf :
        x.append(x[-1] + dt * f(t[-1], x[-1]))
        t.append(t[-1] + dt)
    return t, x


def f(t, x):
    return x


t0, x0 = 0, 1
tf = 5
dt = 0.001

t, x = solve_euler_explicit(f, dt, tf, x0 , t0)
sol = [np.exp(temps) for temps in t]


def g(t, x):
    return np.cos(t)*x


t0, x0 = 0, 1
tf = 5

t1, x1 = solve_euler_explicit(g, 0.001, tf, x0 , t0 = 0)
sol = [np.exp(np.sin((temps))) for temps in t]


def ecart_max(u, f, f_sol, dt, tf, x0, t0 = 0):
    t, x = u(f, dt, tf, x0, t0 = 0)
    e = []
    for i, temp in enumerate(t):
        e.append(abs(x[i] - f_sol(temp)))
    return max(e)

pas = np.arange(0.001, 0.1, 0.001)
f_sol = np.exp

e_11 = []
for dt in pas:
    e_11.append(ecart_max(solve_euler_explicit, f, f_sol, dt, 10, 1))

def solve_heun_explicit(f, dt, tf, x0, t0 = 0):
    t, x = [t0, t0 + dt], [x0]
    i = 0
    while t[-1] < tf:
        x.append(x[-1] + dt/2 * (f(t[i], x[-1]) + f(t[i+1], x[-1] + dt * f(t[i], x[-1]))))
        t.append(t[-1] + dt)
        i += 1
    return t[:-1], x


x0 = 1
tf = 5

t, x = solve_heun_explicit(f, 0.001, tf, x0 , t0 = 0)
sol = [np.exp(temps) for temps in t]

plt.plot(t, x1, color = 'green')
plt.plot(t, sol, color = 'red')
plt.plot(t, x, color = 'blue')

plt.show()


pas = np.arange(0.001, 0.1, 0.001)

e_2 = []
for dt in pas:
    e_2.append(ecart_max(solve_heun_explicit, f, dt, 10, 1))


plt.plot(pas**2, e_2, '+', color = 'purple')
plt.show()