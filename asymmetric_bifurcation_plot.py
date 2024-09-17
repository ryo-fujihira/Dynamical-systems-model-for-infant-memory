import numpy as np
from scipy.integrate import odeint
from ratio_calculation import calc_baseline_ratio, calc_retention_ratio
import matplotlib.pyplot as plt

def func_with_mobile(v, t, alpha, gamma, eta, epsilon, a, b, c_1, c_2, T, w0, W0, tau_u, tau_v, tau_m_aquire):
    dxdt = v[1]
    dvxdt = - (gamma + alpha * (v[0]**2)) * v[1] - (w0**2 + v[4]) * v[0]
    dydt = v[3]
    dvydt = - epsilon * v[3] - (W0**2) * v[2] + c_1 * v[0] + c_2 * np.sin(2 * np.pi * t / T)
    dudt = (a * (v[2]**2) + v[5] * v[2]**2 + eta * v[6] - v[4]) / tau_u
    dvdt = (v[2] ** 2 * v[0] ** 2 - 0.65 * v[2] ** 2 + b * v[5] - v[5] ** 3) / tau_v
    dmdt = (np.tanh(v[4]) - v[6]) / tau_m_aquire
    return [dxdt, dvxdt, dydt, dvydt, dudt, dvdt, dmdt]


def func_without_mobile(v, t, alpha, gamma, eta, epsilon, a, b, c, A, T, w0, W0, tau_u, tau_v, tau_m_decay):
    dxdt = v[1]
    dvxdt = - (gamma + alpha * (v[0]**2)) * v[1] - (w0**2 + v[4]) * v[0]
    dydt = v[3]
    dvydt = - epsilon * v[3] - (W0**2) * v[2] + c * v[0] + A * np.sin(2 * np.pi * t / T)
    dudt = (a * (v[2]**2) + v[5] * v[2]**2 + eta * v[6] - v[4]) / tau_u
    dvdt = (v[2] ** 2 * v[0] ** 2 - 0.65 * v[2] ** 2 + b * v[5] - v[5] ** 3) / tau_v
    dmdt = - v[6] / tau_m_decay
    return [dxdt, dvxdt, dydt, dvydt, dudt, dvdt, dmdt]


def simulation(condition, init, t, manual_b=None):

    if condition == 'stimulation':
        c = 0
        A = 1.2
        eta = 3
        gamma = -0.25
    elif condition == 'interaction':
        c = 2
        A = 0
        eta = 3
        gamma = -0.25
    elif condition == 'baseline':
        c = 0
        A = 0
        eta = 3
        gamma = -0.25
    elif condition == 'no mobile':
        c = 0
        A = 0
        eta = 0
        gamma = -0.25

    alpha = 1
    w0 = 0.6
    W0 = 2.2
    tau_u = 100
    tau_v = 200
    tau_m_aquire = 200
    tau_m_decay = 10000
    a = 2
    b = 14
    epsilon = 0.5
    T = 3

    if condition == 'no mobile':
        tragectory = odeint(func_without_mobile, init, t, args=(alpha, gamma, eta, epsilon, a, b, c, A, T, w0, W0, tau_u, tau_v, tau_m_decay))
    else:
        tragectory = odeint(func_with_mobile, init, t, args=(alpha, gamma, eta, epsilon, a, b, c, A, T, w0, W0, tau_u, tau_v, tau_m_aquire))
    return tragectory, tragectory[-1, :]



t_1 = np.arange(0, 250, 0.1) 
t_2 = np.arange(0, 500, 0.1)

v0 = [1, 0, 0, 0, 0, 0, 0]
sim_base1_1, v1 = simulation('baseline', v0, t_1)
sim_interaction1, v2 = simulation('interaction', v1, t_1)
sim_stimulation1, v3 = simulation('stimulation', v2, t_2)

v0 = [1, 0, 0, 0, 0, 0, 0]
sim_base2_1, v1 = simulation('baseline', v0, t_1)
sim_stimulation2, v2 = simulation('stimulation', v1, t_1)
sim_interaction2, v3 = simulation('interaction', v2, t_2)

fig = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.plot(t_1, sim_base1_1[:, 5], c='red', label='Play → Observe', linewidth=2)
plt.plot(t_1+350, sim_interaction1[:, 5], c='red', linewidth=2)
plt.plot(t_2+700, sim_stimulation1[:, 5], c='red', linewidth=2)

plt.plot(t_1, sim_base2_1[:, 5], c='blue', label='Observe → Play', linewidth=2)
plt.plot(t_1+350, sim_stimulation2[:, 5], c='blue', linewidth=2)
plt.plot(t_2+700, sim_interaction2[:, 5], c='blue', linewidth=2)

plt.ylim([-7.5, 10])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tick_params(labelsize=16)
plt.gca().axes.yaxis.set_ticks([-5, 0, 5])
plt.xticks([0, 300, 650, 1200], color='None')
plt.legend(fontsize=16, loc='upper left')
plt.show()
