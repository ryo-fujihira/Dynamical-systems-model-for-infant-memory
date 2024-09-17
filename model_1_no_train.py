import numpy as np
from scipy.integrate import odeint
from ratio_calculation import calc_baseline_ratio, calc_retention_ratio
import matplotlib.pyplot as plt

def func_with_mobile(v, t, alpha, gamma, eta, epsilon, a, c, A, T, w0, W0, tau_u, tau_m_aquire):
    dxdt = v[1]
    dvxdt = - (gamma + alpha * (v[0]**2)) * v[1] - (w0**2 + v[4]) * v[0]
    dydt = v[3]
    dvydt = - epsilon * v[3] - (W0**2) * v[2] + c * v[0] + A * np.sin(2 * np.pi * t / T)
    dudt = (a * (v[2]**2) + eta * v[5] - v[4]) / tau_u
    dmdt = (np.tanh(v[4]) - v[5]) / tau_m_aquire
    return [dxdt, dvxdt, dydt, dvydt, dudt, dmdt]


def func_without_mobile(v, t, alpha, gamma, eta, epsilon, a, c, A, T, w0, W0, tau_u, tau_m_decay):
    dxdt = v[1]
    dvxdt = - (gamma + alpha * (v[0]**2)) * v[1] - (w0**2 + v[4]) * v[0]
    dydt = v[3]
    dvydt = - epsilon * v[3] - (W0**2) * v[2] + c * v[0] + A * np.sin(2 * np.pi * t / T)
    dudt = (a * (v[2]**2) + eta * v[5] - v[4]) / tau_u
    dmdt =  - v[5] / tau_m_decay
    return [dxdt, dvxdt, dydt, dvydt, dudt, dmdt]


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
    tau_m_aquire = 200
    tau_m_decay = 10000
    a = 4
    epsilon = 0.5
    T = 3
    if condition == 'no mobile':
        tragectory = odeint(func_without_mobile, init, t, args=(alpha, gamma, eta, epsilon, a, c, A, T, w0, W0, tau_u, tau_m_decay))
    else:
        tragectory = odeint(func_with_mobile, init, t, args=(alpha, gamma, eta, epsilon, a, c, A, T, w0, W0, tau_u, tau_m_aquire))
    return tragectory, tragectory[-1, :]


v0 = [1, 0, 0, 0, 0, 0]

t_3min = np.arange(0, 300, 0.1) 
t_9min = np.arange(0, 900, 0.1)
tau_m_decay = 10000
t_1day = np.arange(0, tau_m_decay / 5, 0.1)  # tau_m_decay / 5
t_13days = np.arange(0, 13 * tau_m_decay / 5, 0.1)
t_27days = np.arange(0, 27 * tau_m_decay / 5, 0.1)

sim_base1, v1 = simulation('no mobile', v0, t_3min)
sim_train1, v2 = simulation('no mobile', v1, t_9min)
sim_extinction1, v3 = simulation('no mobile', v2, t_3min)

sim_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_base2, v5 = simulation('no mobile', v4, t_3min)
sim_train2, v6 = simulation('no mobile', v5, t_9min)
sim_extinction2, v7 = simulation('no mobile', v6, t_3min)

# original memory
t_interval = np.arange(0, tau_m_decay*14/5, 0.1)
sim_interval2, v8 = simulation('no mobile', v7, t_interval)
sim_base3, v9 = simulation('baseline', v8, t_3min)
sim_train3, v10 = simulation('interaction', v9, t_9min)
sim_extinction3, v11 = simulation('baseline', v10, t_3min)

# reactivated memory
v0 = [1, 1, 0, 0, 0, 0]

sim_re_base1, v1 = simulation('baseline', v0, t_3min)
sim_re_train1, v2 = simulation('no mobile', v1, t_9min)
sim_re_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_re_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_re_base2, v5 = simulation('baseline', v4, t_3min)
sim_re_train2, v6 = simulation('no mobile', v5, t_9min)
sim_re_extinction2, v7 = simulation('baseline', v6, t_3min)

sim_re_interval2, v8 = simulation('no mobile', v7, t_13days)

sim_re_reactivation, v9 = simulation('stimulation', v8, t_3min)

sim_re_interval3, v10 = simulation('no mobile', v9, t_1day)

sim_re_base3, v11 = simulation('baseline', v10, t_3min)
sim_re_train3, v12 = simulation('interaction', v11, t_9min)
sim_re_extinction3, v13 = simulation('baseline', v12, t_3min)


# plot mean kicking 
b, mean_kick1 = calc_baseline_ratio(sim_base1, np.concatenate([sim_train1, sim_extinction1]), 1200)
b, mean_kick2 = calc_baseline_ratio(sim_base2, np.concatenate([sim_train2, sim_extinction2]), 1200)
b, mean_kick3 = calc_baseline_ratio(sim_base3, np.concatenate([sim_train3, sim_extinction3]), 1200)

b, mean_re_kick1 = calc_baseline_ratio(sim_re_base1, np.concatenate([sim_re_train1, sim_re_extinction1]), 1200)
b, mean_re_kick2 = calc_baseline_ratio(sim_re_base2, np.concatenate([sim_re_train2, sim_re_extinction2]), 1200)
b, mean_re_kick3 = calc_baseline_ratio(sim_re_base3, np.concatenate([sim_re_train3, sim_re_extinction3]), 1200)

fig = plt.figure(figsize=(9, 3.5))
time = [1, 2, 3, 4, 5]

plt.subplot(132)
plt.plot(time, mean_kick2, c='black')
plt.plot(time, mean_re_kick2, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.yaxis.set_ticks([])
plt.xticks([1, 2, 3, 4, 5])
plt.tick_params(labelsize=14)
plt.subplot(131)
plt.plot(time, mean_kick1, c='black', label='No reactivation')
plt.plot(time, mean_re_kick1, c='red', linestyle='--', label='Reactivation')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xticks([1, 2, 3, 4, 5])
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=16)
plt.tick_params(labelsize=14)
plt.subplot(133)
plt.plot(time, mean_kick3, c='black')
plt.plot(time, mean_re_kick3, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.yaxis.set_ticks([])
plt.xticks([1, 2, 3, 4, 5])
plt.tick_params(labelsize=14)

# variation of u, m during the tasks
fig = plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.plot(t_3min, sim_re_base1[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train1[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction1[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base1[:, 5], c='red')
plt.plot(t_9min+300, sim_re_train1[:, 5], c='red')
plt.plot(t_3min+1200, sim_re_extinction1[:, 5], c='red')
plt.plot(t_3min, sim_base1[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train1[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction1[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base1[:, 5], c='black')
plt.plot(t_9min+300, sim_train1[:, 5], c='black')
plt.plot(t_3min+1200, sim_extinction1[:, 5], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.tick_params(labelsize=14)
plt.ylim([-0.5, 6.5])
plt.subplot(132)
plt.plot(t_3min, sim_re_base2[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train2[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction2[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base2[:, 5], c='red')
plt.plot(t_9min+300, sim_re_train2[:, 5], c='red')
plt.plot(t_3min+1200, sim_re_extinction2[:, 5], c='red')
plt.plot(t_3min, sim_base2[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train2[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction2[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base2[:, 5], c='black')
plt.plot(t_9min+300, sim_train2[:, 5], c='black')
plt.plot(t_3min+1200, sim_extinction2[:, 5], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)
plt.ylim([-0.5, 6.5])
plt.subplot(133)
plt.plot(t_3min, sim_re_base3[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train3[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction3[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base3[:, 5], c='red')
plt.plot(t_9min+300, sim_re_train3[:, 5], c='red')
plt.plot(t_3min+1200, sim_re_extinction3[:, 5], c='red')
plt.plot(t_3min, sim_base3[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train3[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction3[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base3[:, 5], c='black')
plt.plot(t_9min+300, sim_train3[:, 5], c='black')
plt.plot(t_3min+1200, sim_extinction3[:, 5], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)
plt.ylim([-0.5, 6.5])
plt.show()

v0 = [1, 1, 0, 0, 0, 0]

sim_base1, v1 = simulation('baseline', v0, t_3min)
sim_train1, v2 = simulation('no mobile', v1, t_9min)
sim_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_base2, v5 = simulation('baseline', v4, t_3min)
sim_train2, v6 = simulation('no mobile', v5, t_9min)
sim_extinction2, v7 = simulation('baseline', v6, t_3min)

sim_interval2, v8 = simulation('no mobile', v7, t_13days)
sim_interval2, v8 = simulation('no mobile', v8, t_3min)

sim_reactivation, v9 = simulation('stimulation', v8, t_3min)
sim_no_reactivation, v10 = simulation('no mobile', v8, t_3min)

sim_interval3_1, v11 = simulation('no mobile', v9, t_3min)
sim_interval3_2, v12 = simulation('no mobile', v10, t_3min)

fig = plt.figure()
plt.plot(t_3min, sim_interval2[:, 4], c='red', linestyle='--')
plt.plot(t_3min+300, sim_reactivation[:, 4], c='red', linestyle='--')
plt.plot(t_3min+600, sim_interval3_1[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_interval2[:, 5], c='red')
plt.plot(t_3min+300, sim_reactivation[:, 5], c='red')
plt.plot(t_3min+600, sim_interval3_1[:, 5], c='red')
plt.plot(t_3min, sim_interval2[:, 4], c='black', linestyle='--')
plt.plot(t_3min+300, sim_no_reactivation[:, 4], c='black', linestyle='--')
plt.plot(t_3min+600, sim_interval3_2[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_interval2[:, 5], c='black')
plt.plot(t_3min+300, sim_no_reactivation[:, 5], c='black')
plt.plot(t_3min+600, sim_interval3_2[:, 5], c='black')
plt.xticks([0, 300, 600, 900], [0, 3, 6, 9])
plt.ylim([-1, 6])
plt.yticks([0, 2, 4, 6])
plt.tick_params(labelsize=18)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()