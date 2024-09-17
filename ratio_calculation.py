import numpy as np


def calc_baseline_ratio(baseline, experimental_data, experimental_time):
    baseline_counts = []
    for i in range(1):
        data = baseline[3000*i:3000*(i+1), 0]
        baseline_count = 0
        for j in range(len(data)-1):
            if (data[j] > 0) and (data[j+1]) <= 0:
                baseline_count += 1
        baseline_counts.append(baseline_count/3)
    counts = []
    time = int(experimental_time / 300)
    for i in range(time):
        data = experimental_data[3000*i:3000*(i+1), 0]
        count = 0
        for j in range(len(data)-1):
            if (data[j] > 0) and (data[j+1]) <= 0:
                count += 1
        counts.append(count/3)

    baseline_ratio = []
    for x in baseline_counts:
        baseline_ratio.append(x / np.mean(baseline_counts))
    for x in counts:
        baseline_ratio.append(x / np.mean(baseline_counts))
    return baseline_ratio, baseline_counts + counts


def calc_1min_baseline_ratio(baseline, experimental_data, experimental_time):
    baseline_counts = []
    for i in range(3):
        data = baseline[1000*i:1000*(i+1), 0]
        baseline_count = 0
        for j in range(len(data)-1):
            if (data[j] > 0) and (data[j+1]) <= 0:
                baseline_count += 1
        baseline_counts.append(baseline_count)
    counts = []
    time = int(experimental_time / 100)
    for i in range(time):
        data = experimental_data[1000*i:1000*(i+1), 0]
        count = 0
        for j in range(len(data)-1):
            if (data[j] > 0) and (data[j+1]) <= 0:
                count += 1
        counts.append(count)

    baseline_ratio = []
    for x in baseline_counts:
        baseline_ratio.append(x / np.mean(baseline_counts))
    for x in counts:
        baseline_ratio.append(x / np.mean(baseline_counts))
    return baseline_ratio, baseline_counts + counts


def calc_retention_ratio(retention, experimental_datas, experimental_time):
    data = retention[:, 0]
    retention_count = 0
    for j in range(len(data) - 1):
        if (data[j] > 0) and (data[j + 1]) <= 0:
            retention_count += 1
    retention_ratios = []
    for experimental_data in experimental_datas:
        data = experimental_data[:, 0]
        count = 0
        for j in range(len(data) - 1):
            if (data[j] > 0) and (data[j + 1]) <= 0:
                count += 1

        retention_ratio = count / retention_count
        retention_ratios.append(retention_ratio)
    return retention_ratios
