from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            x = float(line)
            data.append(x)
    return np.array(data, dtype=float)

data = read_data("A.txt")
x_mean, x_std = norm.fit(data)
print(x_mean)
print(x_std)