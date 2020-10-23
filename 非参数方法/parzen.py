import numpy as np
import math
import matplotlib.pyplot as plt 
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            x = float(line)
            data.append(x)
    return np.array(data, dtype=np.float32)

def window_func(x):
    return 1/np.square(2*math.pi) * np.exp(-x * x / 2)

datas = read_data("A.txt")
n = len(datas)
max_num = np.max(datas)
min_num = np.min(datas) 
h = 1000 #窗口的大小
centers = np.linspace(min_num, max_num, int(np.round((max_num - min_num) / h)) ) #根据窗口大小计算出每个窗口
#利用parzen窗方法，计算概率密度
prob_density = []

for i in range(len(centers)):
    p = 0
    p = 1 / (n * h) * np.sum(window_func((datas - centers[i]) / h)) 
    prob_density.append(p)

plt.plot(centers, prob_density, linewidth=0.5)
#画图，设置线条宽度
#设置图表标题，给坐标轴添加标签
plt.title("probability distribution", fontsize=10)
plt.xlabel("x", fontsize=5)
plt.ylabel("p(x)", fontsize=5) 
#设置刻度标记的大小
plt.tick_params(axis='both', labelsize=8) 
plt.show()