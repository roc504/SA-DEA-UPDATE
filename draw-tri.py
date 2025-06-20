import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']       # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False         # 正确显示负号

# ---------------------
# STEP 1: 读取 CSV，提取摄像头坐标
# ---------------------
df = pd.read_csv("E:\\小论文\\李振滢论文\\code\\Visibility_Matrix.csv")  # 替换为你的实际路径
# img = mpimg.imread("E:\\camera\\draw\\msp2.jpg")
# 只保留每个摄像头（OID_OBSERV）对应的一条坐标记录
camera_coords = df.groupby("OID_OBSERV")[["Xc", "Yc", "Zc"]].first().reset_index()
# STEP 1: 基准点设定
ref_point = camera_coords.iloc[0][['Xc', 'Yc', 'Zc']].values
camera_coords[['Xc', 'Yc', 'Zc']] = camera_coords[['Xc', 'Yc', 'Zc']] - ref_point
x0, y0, z0= -2291947.878,5002300.177,3214828.76 # 左上
x1, y1, z1= -2291548.54,5002346.826	,3215039.805
x0, y0, z0=(x0, y0, z0)- ref_point
x1, y1, z1= (x1, y1, z1)- ref_point
corner1 = [x0, y0, z0]  # 左上
corner2 = [x1, y0, z0]  # 右上
corner3 = [x1, y1, z1]  # 右下
corner4 = [x0, y1, z1]  # 左下

# 生成网格（图像和坐标点形状要对应）
# img_h, img_w = img.shape[:2]
x = np.array([[corner1[0], corner2[0]],
              [corner4[0], corner3[0]]])
y = np.array([[corner1[1], corner2[1]],
              [corner4[1], corner3[1]]])
z = np.array([[corner1[2], corner2[2]],
              [corner4[2], corner3[2]]])


 # 右下（注意 y1 > y2）
# STEP 2: 之后的绘图中就正常传入这些相对坐标

# ---------------------
# STEP 2: 摄像头列表（编号, pitch, yaw）
# ---------------------
cameras =[[339, 82, 342], [344, 82, 354], [183, 21, 354], [186, 68, 57], [334, 47, 134], [92, 23, 12], [275, 43, 47], [117, 79, 235], [86, 51, 79], [1, 33, 171], [80, 27, 130], [244, 41, 312], [33, 6, 8], [313, 3, 336], [43, 61, 298], [147, 26, 231], [311, 82, 205], [355, 33, 116], [280, 12, 296], [162, 52, 334], [110, 36, 60], [15, 50, 1], [319, 37, 330], [114, 74, 94], [138, 27, 5], [69, 52, 37], [293, 52, 44], [152, 37, 68], [306, 36, 141], [234, 66, 226], [79, 28, 131], [325, 25, 12], [340, 75, 149], [292, 90, 151], [200, 71, 267], [338, 77, 202], [222, 30, 13], [219, 55, 141], [347, 63, 15], [159, 67, 70], [279, 32, 15], [30, 78, 16], [118, 73, 232], [215, 37, 1], [157, 53, 105], [345, 70, 118], [66, 52, 211], [256, 36, 277], [164, 23, 298], [137, 21, 311]]

# 参数设置
fov = 100         # 视场角
distance = 60    # 可视距离
resolution = 20  # 圆锥底部边数

# ---------------------
# STEP 3: 绘制锥体函数
# ---------------------
def create_cone_faces(origin, pitch, yaw, fov, distance, resolution):
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    fov_rad = math.radians(fov)

    # 视线方向向量
    dx = math.cos(pitch_rad) * math.cos(yaw_rad)
    dy = math.cos(pitch_rad) * math.sin(yaw_rad)
    dz = -math.sin(pitch_rad)
    direction = np.array([dx, dy, dz])
    direction = direction / np.linalg.norm(direction)

    # 锥底中心点
    center = origin + direction * distance
    radius = distance * math.tan(fov_rad / 2)

    # 构造垂直于 direction 的局部坐标系
    if np.allclose(direction, [0, 0, 1]):
        up = np.array([0, 1, 0])
    else:
        up = np.array([0, 0, 1])
    right = np.cross(direction, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, direction)
    up = up / np.linalg.norm(up)

    # 构造底面圆周点
    circle_pts = []
    for i in range(resolution):
        angle = 2 * np.pi * i / resolution
        point = center + radius * (math.cos(angle) * right + math.sin(angle) * up)
        circle_pts.append(point)

    # 构造锥体侧面三角形
    faces = []
    for i in range(resolution):
        p1 = circle_pts[i]
        p2 = circle_pts[(i + 1) % resolution]
        faces.append([origin, p1, p2])
    return faces


# ---------------------
# STEP 4: 可视化锥形视野图
# ---------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# img = img / 255.0
# 贴图图像
# ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False)

for cam in cameras:
    oid, pitch, yaw = cam
    match = camera_coords[camera_coords["OID_OBSERV"] == oid]
    if match.empty:
        print(f"摄像头编号 {oid} 未在CSV中找到，跳过")
        continue
    x, y, z = match.iloc[0][["Xc", "Yc", "Zc"]].values
    origin = np.array([x, y, z])

    faces = create_cone_faces(origin, pitch, yaw, fov, distance, resolution)
    color = plt.cm.viridis(random.random())
    cone = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='gray')
    ax.add_collection3d(cone)
    ax.scatter(*origin, color='black', s=5)

# 设置图形
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('摄像头三维视域图')
ax.set_box_aspect([1, 1, 0.6])
plt.tight_layout()
plt.show()
