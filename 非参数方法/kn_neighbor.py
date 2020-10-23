import numpy as np
import math
import matplotlib.pyplot as plt 
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            x = float(line)
            data.append(x)
    data.sort()
    return np.array(data, dtype=np.float32)

def window_func(x):
    return 1/np.square(2*math.pi) * np.exp(-x * x / 2)

datas = read_data("A.txt")
n = len(datas)
k = 300 #近邻的个数

#计算每个点的k+1近邻
#因为自己到自己的距离为0.所以多计算一个近邻。在之后计算概率密度时排除
distances = np.abs(datas.reshape((n, 1)) - datas.reshape((1, n)))

k_neighbors = np.argpartition(distances, k + 1)[ :, :k+1] #k_neighbors[i]：data[i]的k+1个近邻的编号

#利用kn近邻，计算概率密度
prob_density = []

for i in range(n):
    min_pos = datas[i]
    max_pos = datas[i]
    for idx in k_neighbors[i]:
        if idx == i:
            continue
        if datas[idx] > max_pos:
            max_pos = datas[idx]
        if datas[idx] < min_pos:
            min_pos = datas[idx]
    h = max_pos - min_pos
    p = k / (n * h)
    prob_density.append(p) 



#画图，设置线条宽度
plt.plot(datas, prob_density, linewidth=0.5)
#设置图表标题，给坐标轴添加标签
plt.title("probability distribution", fontsize=15)
plt.xlabel("x", fontsize=10)
plt.ylabel("p(x)", fontsize=10) 
#设置刻度标记的大小
plt.tick_params(axis='both', labelsize=10) 
plt.show()