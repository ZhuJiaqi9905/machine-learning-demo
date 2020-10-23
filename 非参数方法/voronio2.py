from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt 

def get_points_xy(points):
    point_x = []
    point_y = []
    for p in points:
        point_x.append(p[0])
        point_y.append(p[1])
    return point_x, point_y
points = [[0, 1], [1, 1], [1, 3], [4, 3], [7, 0], [7, 3], [8, 1], [8, 2], [8, 3], [9, 0],
        [0, 5], [0, 6], [1, 5], [1, 8], [2, 6], [2, 7], [2, 8], [3, 6], [4, 6], [4, 8],
        [4, 5], [6, 4], [6, 8], [7, 4], [8, 4], [9, 4], [9, 6], [9, 7], [4, 9], [6, 7]
        ]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

res = [[0, 1], [1, 3], [4, 3], [7, 3], [8, 3], [8, 2], [9, 0],
        [0 ,5], [1, 5], [3, 6], [4, 6], [2, 8], [4, 8],[1, 8],
        [4, 5], [4, 9], [6, 4], [7, 4], [8, 4], [9, 4], [6, 7], [6, 8]
]

remain = [[1, 1], [7, 0], [8, 1], 
        [0, 6], [2, 6], [2, 7], 
        [9, 6], [9, 7]
            ]
#总数据点
points1_x, points1_y = get_points_xy(points[0: 10])
points2_x, points2_y = get_points_xy(points[10: 20])
points3_x, points3_y = get_points_xy(points[20: 30])
#能用用画voronio图的点
res1_x, res1_y = get_points_xy(res[0: 7])
res2_x, res2_y = get_points_xy(res[7: 14])
res3_x, res3_y = get_points_xy(res[14: 22])
#剩下的点
remain1_x, remain1_y = get_points_xy(remain[0: 3])
remain2_x, remain2_y = get_points_xy(remain[3: 6])
remain3_x, remain3_y = get_points_xy(remain[6: 8])



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