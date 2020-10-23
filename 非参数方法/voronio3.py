from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np
import math
import random 
data = [[0, 1, 1], [1, 1, 1], [1, 3, 1], [4, 3, 1], [7, 0, 1], [7, 3, 1], [8, 1, 1], [8, 2, 1], [8, 3, 1], [9, 0, 1],
        [0, 5, 2], [0, 6, 2], [1, 5, 2], [1, 8, 2], [2, 6, 2], [2, 7, 2], [2, 8, 2], [3, 6, 2], [4, 6, 2], [4, 8, 2],
        [4, 5, 3], [6, 4, 3], [6, 8, 3], [7, 4, 3], [8, 4, 3], [9, 4, 3], [9, 6, 3], [9, 7, 3], [4, 9, 3], [6, 7, 3]
        ]
label = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

dataA = []
dataB = data.copy()

choice = list(range(0, 30))
random.shuffle(choice)


def test_neighbor(data):
    label = []
    for d in data:
        label.append(d[2])
    distance = []
    for i in range(len(data)):
        dis = []
        for j in range(len(data)):
            if i == j:
                dis.append(100000)
                continue
            d = math.sqrt( (data[i][0] - data[j][0]) **2 + (data[i][1] - data[j][1]) **2 )
            dis.append(d)
        distance.append(dis)
    distance = np.array(distance)
    label = np.array(label)
    idx = np.argmin(distance, axis=1) 
    predict = label[ idx ]
    corr = np.sum((predict == label))
    return corr


def get_points_xy(points):
    point_x = []
    point_y = []
    for p in points:
        point_x.append(p[0])
        point_y.append(p[1])
    return point_x, point_y

def get_diff_points(data):
    p1_x = []
    p1_y = []
    p2_x = []
    p2_y = []
    p3_x = []
    p3_y = []

    for d in data:
        if d[2] == 1:
            p1_x.append(d[0])
            p1_y.append(d[1])
        elif d[2] == 2:
            p2_x.append(d[0])
            p2_y.append(d[1])
        else:
            p3_x.append(d[0])
            p3_y.append(d[1])
    return p1_x, p1_y, p2_x, p2_y, p3_x, p3_y

#算法开始
for i in choice:
    ele = data[i]
    dataA.append(ele)
    dataB.remove(ele) 
    corr = test_neighbor(dataB) 

    if corr == len(dataB):
        dataA.remove(ele)
        dataB.append(ele) 

#res是用于画图的点
res = dataA
res1_x, res1_y, res2_x, res2_y, res3_x, res3_y = get_diff_points(res) 

#remain是被压缩的点
remain = []
for d in data:
    flag = False
    for r in res:
        if r[0] == d[0] and r[1] == d[1]:
            flag = True 
    if not flag:
        
        remain.append(d)

remain1_x, remain1_y, remain2_x, remain2_y, remain3_x, remain3_y = get_diff_points(remain)

for r in res:
    del r[2]
vor = Voronoi(res)
fig = voronoi_plot_2d(vor, show_vertices=False, show_points = False,  line_colors='black',line_width=0.5, line_alpha=0.6, point_size=5)
plt.xlim(-2, 13) #设置横纵坐标范围
plt.ylim(-2, 13)

plt.scatter(res1_x, res1_y, c="blue", linewidths=1, alpha=1)
plt.scatter(res2_x, res2_y, c="yellow", linewidths=1, alpha=1)
plt.scatter(res3_x, res3_y, c="red", linewidths=1, alpha=1)


plt.scatter(remain1_x, remain1_y, c = 'blue', linewidths=0.5, alpha=0.2)
plt.scatter(remain2_x, remain2_y, c = 'yellow', linewidths=0.5, alpha=0.2)
plt.scatter(remain3_x, remain3_y, c = 'red', linewidths=0.5, alpha=0.2)
plt.show()