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
    b = 12
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

sim_base1, v1 = simulation('baseline', v0, t_3min)
sim_train1, v2 = simulation('interaction', v1, t_9min)
sim_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_base2, v5 = simulation('baseline', v4, t_3min)
sim_train2, v6 = simulation('interaction', v5, t_9min)
sim_extinction2, v7 = simulation('baseline', v6, t_3min)

fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(111)

# original memory
retention_intervals = [1, 2, 3, 4, 5, 7, 8, 14, 28]
retentions = []
v8m = []
v8u = []
v8v = []
for i in range(len(retention_intervals)):
    t_interval = np.arange(0, tau_m_decay*retention_intervals[i]/5, 0.1)
    sim_interval2, v8 = simulation('no mobile', v7, t_interval)
    print(i, v8)
    retention, retention_last = simulation('baseline', v8, t_3min)
    retentions.append(retention)
    v8m.append(v8[-1])
    v8u.append(v8[-3])
    v8v.append(v8[-2])

    if i == 7:
        sim_base3, v9 = simulation('baseline', v8, t_3min)
        sim_train3, v10 = simulation('interaction', v9, t_9min)
        sim_extinction3, v11 = simulation('baseline', v10, t_3min)

retention_ratio = calc_retention_ratio(sim_extinction2, retentions, 300)
print(retention_ratio)
time = [1, 2, 3, 4, 5, 7, 8, 14, 28]
plt.plot(time, retention_ratio, marker='o', label='Original memory', c='black')

# reactivated memory
v0 = [1, 1, 0, 0, 0, 0, 0]

sim_re_base1, v1 = simulation('baseline', v0, t_3min)
sim_re_train1, v2 = simulation('interaction', v1, t_9min)
sim_re_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_re_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_re_base2, v5 = simulation('baseline', v4, t_3min)
sim_re_train2, v6 = simulation('interaction', v5, t_9min)
sim_re_extinction2, v7 = simulation('baseline', v6, t_3min)

sim_re_interval2, v8 = simulation('no mobile', v7, t_13days)

sim_re_reactivation, v9 = simulation('stimulation', v8, t_3min)

react_retention_intervals = [1, 3, 6, 9, 15]
react_retentions = []
v10m = []
v10u = []
v10v = []
for i in range(len(react_retention_intervals)):
    t_interval = np.arange(0, tau_m_decay*react_retention_intervals[i]/5, 0.1)
    sim_re_interval3, v10 = simulation('no mobile', v9, t_interval)
    retention, retention_last = simulation('baseline', v10, t_3min)
    react_retentions.append(retention)
    v10m.append(v10[-1])
    v10u.append(v10[-3])
    v10v.append(v10[-2])

    if i == 0:
        sim_re_base3, v11 = simulation('baseline', v10, t_3min)
        sim_re_train3, v12 = simulation('interaction', v11, t_9min)
        sim_re_extinction3, v13 = simulation('baseline', v12, t_3min)
    
react_retention_ratio = calc_retention_ratio(sim_re_extinction2, react_retentions, 300)
print(react_retention_ratio)
time = [14, 16, 19, 22, 28]
plt.plot(time, react_retention_ratio, marker='o', label='Reactivated memory', c='red')

# reactivated memory 2
v0 = [1, 0.5, 0, 0, 0, 0, 0]

sim_re2_base1, v1 = simulation('baseline', v0, t_3min)
sim_re2_train1, v2 = simulation('interaction', v1, t_9min)
sim_re2_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_re2_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_re2_base2, v5 = simulation('baseline', v4, t_3min)
sim_re2_train2, v6 = simulation('interaction', v5, t_9min)
sim_re2_extinction2, v7 = simulation('baseline', v6, t_3min)

sim_re2_interval2, v8 = simulation('no mobile', v7, t_27days)

sim_re2_reactivation, v9 = simulation('stimulation', v8, t_3min)

sim_re2_interval3, v10 = simulation('no mobile', v9, t_1day)
sim_re2_base3, v11 = simulation('baseline', v10, t_3min)
sim_re2_train3, v12 = simulation('interaction', v11, t_9min)
sim_re2_extinction3, v13 = simulation('baseline', v12, t_3min)
react_retention_ratio_2nd = calc_retention_ratio(sim_re2_extinction2, [sim_re2_base3], 300)
plt.scatter([28], [react_retention_ratio_2nd], c='red')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', fontsize=18)
plt.tick_params(labelsize=16)
# plt.ylim([0.65, 1.1])
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
plt.ylim([0.25, 0.75])
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
time = [1, 2, 3, 4, 5, 7, 8, 14, 28]
ax1.plot(time, v8m, c='black', label='$u_{mem}$ original', marker='o')
ax1.plot(time, v8u, c='black', linestyle='--', label='$\it{u}$ original', marker='o')
ax2.plot(time, v8v, c='black', linestyle='dotted', label='$\it{v}$ original', marker='o')
time = [14, 16, 19, 22, 28]
ax1.plot(time, v10m, c='red', label='$u_{mem}$ reactivated', marker='o')
ax1.plot(time, v10u, c='red', linestyle='--', label='$\it{u}$ reactivated', marker='o')
ax2.plot(time, v10v, c='red', linestyle='dotted', label='$\it{v}$ reactivated', marker='o')
# plt.legend(bbox_to_anchor=(1, 1), fontsize=16)
ax1.tick_params(labelsize=16)
ax2.tick_params(labelsize=16)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax1.set_ylim([-0.2, 1.0])
ax2.set_yticks([0, 1, 2, 3, 4])
ax1.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# plot mean kicking 
b, mean_kick1 = calc_baseline_ratio(sim_base1, np.concatenate([sim_train1, sim_extinction1]), 1200)
b, mean_kick2 = calc_baseline_ratio(sim_base2, np.concatenate([sim_train2, sim_extinction2]), 1200)
b, mean_kick3 = calc_baseline_ratio(sim_base3, np.concatenate([sim_train3, sim_extinction3]), 1200)

b, mean_re_kick1 = calc_baseline_ratio(sim_re_base1, np.concatenate([sim_re_train1, sim_re_extinction1]), 1200)
b, mean_re_kick2 = calc_baseline_ratio(sim_re_base2, np.concatenate([sim_re_train2, sim_re_extinction2]), 1200)
b, mean_re_kick3 = calc_baseline_ratio(sim_re_base3, np.concatenate([sim_re_train3, sim_re_extinction3]), 1200)

b, mean_re2_kick1 = calc_baseline_ratio(sim_re2_base1, np.concatenate([sim_re2_train1, sim_re2_extinction1]), 1200)
b, mean_re2_kick2 = calc_baseline_ratio(sim_re2_base2, np.concatenate([sim_re2_train2, sim_re2_extinction2]), 1200)
b, mean_re2_kick3 = calc_baseline_ratio(sim_re2_base3, np.concatenate([sim_re2_train3, sim_re2_extinction3]), 1200)

fig = plt.figure(figsize=(9, 7))
time = [1, 2, 3, 4, 5]
plt.subplot(231)
plt.plot(time, mean_kick1, c='black', label='No reactivation')
plt.plot(time, mean_re_kick1, c='red', linestyle='--', label='Reactivation')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().axes.xaxis.set_ticks([])
plt.legend(bbox_to_anchor=(0, 1), loc='lower left', fontsize=16)
plt.tick_params(labelsize=14)
plt.subplot(232)
plt.plot(time, mean_kick2, c='black')
plt.plot(time, mean_re_kick2, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)
plt.subplot(233)
plt.plot(time, mean_kick3, c='black')
plt.plot(time, mean_re_kick3, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.xaxis.set_ticks([])
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)
plt.subplot(234)
plt.plot(time, mean_kick1, c='black')
plt.plot(time, mean_re2_kick1, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xticks([1, 2, 3, 4, 5])
plt.tick_params(labelsize=14)
plt.subplot(235)
plt.plot(time, mean_kick2, c='black')
plt.plot(time, mean_re2_kick2, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.yaxis.set_ticks([])
plt.xticks([1, 2, 3, 4, 5])
plt.tick_params(labelsize=14)
plt.subplot(236)
plt.plot(time, mean_kick3, c='black')
plt.plot(time, mean_re2_kick3, c='red', linestyle='--')
plt.ylim([0, 46])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().axes.yaxis.set_ticks([])
plt.xticks([1, 2, 3, 4, 5])
plt.tick_params(labelsize=14)
plt.tight_layout()


# variation of u, v, m during the tasks
fig = plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.plot(t_3min, sim_re_base1[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train1[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction1[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base1[:, 5], c='red', linestyle='dotted')
plt.plot(t_9min+300, sim_re_train1[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min+1200, sim_re_extinction1[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min, sim_re_base1[:, 6], c='red')
plt.plot(t_9min+300, sim_re_train1[:, 6], c='red')
plt.plot(t_3min+1200, sim_re_extinction1[:, 6], c='red')
plt.plot(t_3min, sim_base1[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train1[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction1[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base1[:, 5], c='black', linestyle='dotted')
plt.plot(t_9min+300, sim_train1[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min+1200, sim_extinction1[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min, sim_base1[:, 6], c='black')
plt.plot(t_9min+300, sim_train1[:, 6], c='black')
plt.plot(t_3min+1200, sim_extinction1[:, 6], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.tick_params(labelsize=14)
plt.subplot(132)
plt.plot(t_3min, sim_re_base2[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train2[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction2[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base2[:, 5], c='red', linestyle='dotted')
plt.plot(t_9min+300, sim_re_train2[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min+1200, sim_re_extinction2[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min, sim_re_base2[:, 6], c='red')
plt.plot(t_9min+300, sim_re_train2[:, 6], c='red')
plt.plot(t_3min+1200, sim_re_extinction2[:, 6], c='red')
plt.plot(t_3min, sim_base2[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train2[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction2[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base2[:, 5], c='black', linestyle='dotted')
plt.plot(t_9min+300, sim_train2[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min+1200, sim_extinction2[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min, sim_base2[:, 6], c='black')
plt.plot(t_9min+300, sim_train2[:, 6], c='black')
plt.plot(t_3min+1200, sim_extinction2[:, 6], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)
plt.subplot(133)
plt.plot(t_3min, sim_re_base3[:, 4], c='red', linestyle='--')
plt.plot(t_9min+300, sim_re_train3[:, 4], c='red', linestyle='--')
plt.plot(t_3min+1200, sim_re_extinction3[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_re_base3[:, 5], c='red', linestyle='dotted')
plt.plot(t_9min+300, sim_re_train3[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min+1200, sim_re_extinction3[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min, sim_re_base3[:, 6], c='red')
plt.plot(t_9min+300, sim_re_train3[:, 6], c='red')
plt.plot(t_3min+1200, sim_re_extinction3[:, 6], c='red')
plt.plot(t_3min, sim_base3[:, 4], c='black', linestyle='--')
plt.plot(t_9min+300, sim_train3[:, 4], c='black', linestyle='--')
plt.plot(t_3min+1200, sim_extinction3[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_base3[:, 5], c='black', linestyle='dotted')
plt.plot(t_9min+300, sim_train3[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min+1200, sim_extinction3[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min, sim_base3[:, 6], c='black')
plt.plot(t_9min+300, sim_train3[:, 6], c='black')
plt.plot(t_3min+1200, sim_extinction3[:, 6], c='black')
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.xticks([0, 300, 600, 900, 1200, 1500], [0, 3, 6, 9, 12, 15])
plt.gca().axes.yaxis.set_ticks([])
plt.tick_params(labelsize=14)

plt.show()


t_30s = np.arange(0, 50, 0.1) 
fig = plt.figure(figsize=(10, 7))
plt.rcParams["font.size"] = 14
plt.subplot(231)
plt.plot(t_30s, sim_base1[1500:2000, 0])
plt.plot(t_30s, sim_base1[1500:2000, 2])
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(232)
plt.plot(t_30s, sim_train2[6000:6500, 0])
plt.plot(t_30s, sim_train2[6000:6500, 2])
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.yticks([-1, -0.5, 0, 0.5, 1], [])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(233)
plt.plot(t_30s, sim_extinction2[1500:2000, 0])
plt.plot(t_30s, sim_extinction2[1500:2000, 2])
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.yticks([-1, -0.5, 0, 0.5, 1], [])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(234)
plt.plot(t_30s, sim_base3[1500:2000, 0])
plt.plot(t_30s, sim_base3[1500:2000, 2])
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(235)
plt.plot(t_30s, sim_re_reactivation[1500:2000, 2], c='tab:orange')
plt.plot(t_30s, sim_re_reactivation[1500:2000, 0], c='tab:blue')
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.yticks([-1, -0.5, 0, 0.5, 1], [])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(236)
plt.plot(t_30s, sim_re_base3[-500:, 0])
plt.plot(t_30s, sim_re_base3[-500:, 2])
plt.xticks([0, 25, 50], [0, 0.25, 0.5])
plt.yticks([-1, -0.5, 0, 0.5, 1], [])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

v0 = [1, 1, 0, 0, 0, 0, 0]

sim_base1, v1 = simulation('baseline', v0, t_3min)
sim_train1, v2 = simulation('interaction', v1, t_9min)
sim_extinction1, v3 = simulation('baseline', v2, t_3min)

sim_interval1, v4 = simulation('no mobile', v3, t_1day)

sim_base2, v5 = simulation('baseline', v4, t_3min)
sim_train2, v6 = simulation('interaction', v5, t_9min)
sim_extinction2, v7 = simulation('baseline', v6, t_3min)

sim_interval2, v8 = simulation('no mobile', v7, t_13days)
sim_interval2, v8 = simulation('no mobile', v8, t_3min)

sim_reactivation, v9 = simulation('stimulation', v8, t_3min)
sim_no_reactivation, v10 = simulation('no mobile', v8, t_3min)

sim_interval3_1, v11 = simulation('no mobile', v9, t_3min)
sim_interval3_2, v12 = simulation('no mobile', v10, t_3min)

print(np.mean(sim_base1[:, 0]**2))
print(np.mean(sim_train1[:, 0]**2))
print(np.mean(sim_extinction1[:, 0]**2))
fig = plt.figure()
plt.plot(t_3min, sim_interval2[:, 4], c='red', linestyle='--')
plt.plot(t_3min+300, sim_reactivation[:, 4], c='red', linestyle='--')
plt.plot(t_3min+600, sim_interval3_1[:, 4], c='red', linestyle='--')
plt.plot(t_3min, sim_interval2[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min+300, sim_reactivation[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min+600, sim_interval3_1[:, 5], c='red', linestyle='dotted')
plt.plot(t_3min, sim_interval2[:, 6], c='red')
plt.plot(t_3min+300, sim_reactivation[:, 6], c='red')
plt.plot(t_3min+600, sim_interval3_1[:, 6], c='red')
plt.plot(t_3min, sim_interval2[:, 4], c='black', linestyle='--')
plt.plot(t_3min+300, sim_no_reactivation[:, 4], c='black', linestyle='--')
plt.plot(t_3min+600, sim_interval3_2[:, 4], c='black', linestyle='--')
plt.plot(t_3min, sim_interval2[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min+300, sim_no_reactivation[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min+600, sim_interval3_2[:, 5], c='black', linestyle='dotted')
plt.plot(t_3min, sim_interval2[:, 6], c='black')
plt.plot(t_3min+300, sim_no_reactivation[:, 6], c='black')
plt.plot(t_3min+600, sim_interval3_2[:, 6], c='black')
plt.xticks([0, 300, 600, 900], [0, 3, 6, 9])
plt.ylim([-1, 6])
plt.yticks([0, 2, 4, 6])
plt.tick_params(labelsize=18)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
