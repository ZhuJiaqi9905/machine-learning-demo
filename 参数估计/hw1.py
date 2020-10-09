
import numpy as np
import math
import matplotlib.pyplot as plt 
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            x = float(line)
            data.append(x)
    return np.array(data, dtype=float)
def get_prob_distribution(x, theta, sigma):
    p = np.exp(-np.square(np.log(x) - theta) / (2 * np.square(sigma))) / (sigma * x * np.sqrt(2* math.pi))
    return p
data = read_data("A.txt")
n = float(data.size)
#计算参数
theta = np.sum(np.log(data)) / n
sigma = np.sqrt(np.sum(np.square((np.log(data) - theta))) / n )
print("the theta is ", theta)
print("the sigma is ", sigma)
x = np.linspace(1, 90000, num=90000-1)
y = get_prob_distribution(x, theta, sigma)
plt.plot(x, y, linewidth=2)
#画图，设置线条宽度
#设置图表标题，给坐标轴添加标签
plt.title("probability distribution", fontsize=15)
plt.xlabel("x", fontsize=10)
plt.ylabel("p(x)", fontsize=10) 
#设置刻度标记的大小
plt.tick_params(axis='both', labelsize=10) 
plt.show()