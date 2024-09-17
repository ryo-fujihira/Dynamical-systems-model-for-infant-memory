import numpy as np
from scipy.integrate import odeint
from ratio_calculation import calc_baseline_ratio, calc_retention_ratio
import matplotlib.pyplot as plt

def func_with_mobile(v, t, alpha, gamma, eta, epsilon, a, b, c, A, T, w0, W0, tau_u, tau_v, tau_m_aquire):
    dxdt = v[1]
    dvxdt = - (gamma + alpha * (v[0]**2)) * v[1] - (w0**2 + v[4]) * v[0]
    dydt = v[3]
    dvydt = - epsilon * v[3] - (W0**2) * v[2] + c * v[0] + A * np.sin(2 * np.pi * t / T)
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
        gamma = 0.25
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


v0 = [1, 0, 0, 0, 0, 0, 0]

t_3min = np.arange(0, 300, 0.1) 
t_9min = np.arange(0, 900, 0.1)
tau_m_decay = 10000
t_1day = np.arange(0, tau_m_decay / 5, 0.1)  # tau_m_decay / 5
t_13days = np.arange(0, 13 * tau_m_decay / 5, 0.1)
t_27days = np.arange(0, 27 * tau_m_decay / 5, 0.1)

# reactivated memory
v0 = [1, 1, 0, 0, 0, 0, 0]
react_retentions = []

sim_re_base1, v1 = simulation('baseline', v0, t_3min)
sim_re_train1, v2 = simulation('interaction', v1, t_9min)
sim_re_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_re_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_re_base2, v5 = simulation('baseline', v4, t_3min)
sim_re_train2, v6 = simulation('interaction', v5, t_9min)
sim_re_extinction2, v7 = simulation('baseline', v6, t_3min)

minimum_durations = []
durations = [40 + 20*i for i in range(6)]
days = [6, 9, 13, 16, 20, 27]
for i in range(6):
    sim_re_interval2, v8 = simulation('no mobile', v7, np.arange(0, days[i] * tau_m_decay / 5, 0.1))
    react_retentions = []
    for t in durations:
        sim_re_reactivation, v9 = simulation('stimulation', v8, np.arange(0, t, 0.1))

        sim_re_interval3, v10 = simulation('no mobile', v9, t_1day)
        retention, retention_last = simulation('baseline', v10, t_3min)
        react_retentions.append(retention)
    react_retention_ratio = calc_retention_ratio(sim_re_extinction2, react_retentions, 300)
    print(react_retention_ratio)
    for j, ratio in enumerate(react_retention_ratio):
        if ratio >=0.5:
            print(durations[j])
            minimum_durations.append(durations[j])
            break


fig = plt.figure(figsize=(11, 5))
plt.plot(days, minimum_durations, marker='o', c='black')
plt.tick_params(labelsize=16)
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

'''
fig = plt.figure(figsize=(11, 5))
plt.plot([6, 13, 17, 27], [7.5, 120, 180, 180], c='grey', linestyle=':')
plt.plot([6, 13, 20, 27], [7.5, 120, 180, 180], marker='o', c='black')
plt.tick_params(labelsize=16)
plt.yticks([0, 60, 120, 180])
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
'''
plt.show()
