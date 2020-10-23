from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt 

def get_points_xy(points):
    point_x = []
    point_y = []
    for p in points:
        point_x.append(p[0])
        point_y.append(p[1])
    return point_x, point_y
data = [[0, 1], [1, 1], [1, 3], [4, 3], [7, 0], [7, 3], [8, 1], [8, 2], [8, 3], [9, 0],
        [0, 5], [0, 6], [1, 5], [1, 8], [2, 6], [2, 7], [2, 8], [3, 6], [4, 6], [4, 8],
        [4, 5], [6, 4], [6, 8], [7, 4], [8, 4], [9, 4], [9, 6], [9, 7], [4, 9], [6, 7]
        ]
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

points = data.copy()
points1_x, points1_y = get_points_xy(points[0: 10])
points2_x, points2_y = get_points_xy(points[10: 20])
points3_x, points3_y = get_points_xy(points[20: 30])

vor = Voronoi(points)
fig = voronoi_plot_2d(vor, show_vertices=False, show_points = False,  line_colors='black',line_width=0.5, line_alpha=0.6, point_size=5)
plt.xlim(-2, 13) #设置横纵坐标范围
plt.ylim(-2, 13)
plt.scatter(points1_x, points1_y, c="blue", linewidths=1)
plt.scatter(points2_x, points2_y, c="yellow", linewidths=1)
plt.scatter(points3_x, points3_y, c="red", linewidths=1)

plt.show()