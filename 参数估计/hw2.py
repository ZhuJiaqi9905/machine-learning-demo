
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
def get_prob_distribution(x, mu, sigma):
    p = (1 / (np.sqrt(2*math.pi) * sigma)) * np.exp(-1/2 * np.square((x - mu) / sigma))
    return p
data = read_data("A.txt")
n = float(data.size)
mu = np.sum(data) / n
sigma = np.sqrt(np.sum(np.square((data - mu))) / n )
print("the mean is ", mu)
print("the standard deviation is ", sigma)
x = np.linspace(-100000, 200000, num=300000)
y = get_prob_distribution(x, mu, sigma)
plt.plot(x, y, linewidth=2)
#画图，设置线条宽度
#设置图表标题，给坐标轴添加标签
plt.title("probability distribution", fontsize=15)
plt.xlabel("x", fontsize=10)
plt.ylabel("p(x)", fontsize=10) 
#设置刻度标记的大小
plt.tick_params(axis='both', labelsize=10) 
plt.show()

#画出数据对应的频率直方图
plt.hist(data, bins=500, facecolor="blue")
plt.xlabel("x", fontsize=10)
plt.ylabel("frequency", fontsize=10)
#设置刻度标记的大小
plt.tick_params(axis='both', labelsize=10) 
plt.show()