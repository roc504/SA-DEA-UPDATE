import pandas as pd
import numpy as np
import math
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl


# ----------------------------
# 参数设置
# ----------------------------
# 摄像头视野半角（度），假设为35°
fov_half_deg = 150
fov_half_rad = math.radians(fov_half_deg)

# ----------------------------
# 读取 CSV 数据
# ----------------------------
# CSV 文件 "Visibility_Matrix.csv" 包含字段：
# OID_OBSERV, OID_TARGET, weight, Xt, Yt, Zt, Xc, Yc, Zc, distance_m
df = pd.read_csv('E:\\camera\\data\\Visibility_Matrix.csv')

# ----------------------------
# 定义摄像头组合（示例：25个摄像头，以下只给出部分示例数据）
# 每个摄像头表示为 [OID_OBSERV, 俯仰角, 偏航角]
cameras =[[339, 82, 342], [344, 82, 354], [183, 21, 354], [186, 68, 57], [334, 7, 134], [92, 23, 12], [275, 3, 47], [117, 79, 235], [86, 11, 79], [1, 33, 171], [80, 27, 130], [244, 41, 312], [33, 6, 8], [313, 3, 336], [43, 61, 298], [147, 26, 231], [311, 82, 205], [355, 33, 116], [280, 12, 296], [162, 52, 334], [110, 36, 60], [15, 50, 1], [319, 37, 330], [114, 74, 94], [138, 27, 5], [69, 2, 37], [293, 52, 44], [152, 37, 68], [306, 36, 141], [234, 66, 226], [79, 28, 131], [325, 25, 12], [340, 75, 149], [292, 90, 151], [200, 71, 267], [338, 77, 202], [222, 10, 13], [219, 55, 141], [347, 63, 15], [159, 67, 70], [279, 32, 15], [30, 78, 16], [118, 73, 232], [215, 37, 1], [157, 53, 105], [345, 70, 118], [66, 52, 211], [256, 36, 277], [164, 13, 298], [137, 21, 311]]# ----------------------------
# 初始化字典，存储每个目标点的覆盖次数（同一摄像头对同一目标只算一次）
coverage_count = {}  # key: OID_TARGET, value: 覆盖次数
# 记录每个目标点被覆盖的摄像头偏航角（用于计算视角分散度），注意：仅取水平（偏航角）部分
target_cameras = {}  # key: OID_TARGET, value: list of camera yaw angles in radians

# ----------------------------
# 对每个摄像头判断其可视目标点
# ----------------------------
for cam in cameras:
    cam_id, pitch_deg, yaw_deg = cam
    # 根据题目要求：正的俯仰角表示摄像头往下看，
    # 因此视向向量计算时：v = [cos(pitch)*cos(yaw), cos(pitch)*sin(yaw), -sin(pitch)]
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    view_vector = np.array([
        math.cos(pitch_rad) * math.cos(yaw_rad),
        math.cos(pitch_rad) * math.sin(yaw_rad),
        -math.sin(pitch_rad)
    ])

    # 筛选出 CSV 中该摄像头的记录
    cam_df = df[df['OID_OBSERV'] == cam_id]
    # 使用集合避免同一摄像头对同一目标重复计数
    seen_targets = set()

    for idx, row in cam_df.iterrows():
        cam_coords = np.array([row['Xc'], row['Yc'], row['Zc']])
        target_coords = np.array([row['Xt'], row['Yt'], row['Zt']])
        vec = target_coords - cam_coords
        norm_vec = np.linalg.norm(vec)
        if norm_vec == 0:
            continue
        vec_unit = vec / norm_vec

        # 计算视向与目标方向的夹角
        dot_val = np.dot(view_vector, vec_unit)
        dot_val = max(min(dot_val, 1.0), -1.0)
        angle = math.acos(dot_val)

        # 如果夹角小于视野半角，则目标点在视域内
        if angle <= fov_half_rad:
            target_id = row['OID_TARGET']
            seen_targets.add(target_id)
            # 记录摄像头的偏航角（水平视角），用于后续计算
            if target_id not in target_cameras:
                target_cameras[target_id] = []
            target_cameras[target_id].append(yaw_rad)

    # 对该摄像头覆盖到的每个目标点计数一次
    for target_id in seen_targets:
        coverage_count[target_id] = coverage_count.get(target_id, 0) + 1

# ----------------------------
# 构建唯一目标点 DataFrame
# ----------------------------
# 从 CSV 中获取所有目标点的 OID_TARGET, weight, Xt, Yt
unique_targets = df[['OID_TARGET', 'weight', 'Xt', 'Yt']].drop_duplicates(subset='OID_TARGET').copy()
# 对于未被覆盖的目标点，其覆盖次数设为 0
unique_targets['coverage_count'] = unique_targets['OID_TARGET'].apply(lambda x: coverage_count.get(x, 0))

# ----------------------------
# 计算六个指标
# ----------------------------

# (1) Weighted Coverage Degree (WCD)
#     = (∑_{t in covered} w(t)) / (∑_{t in all} w(t))
total_weight = unique_targets['weight'].sum()
covered_targets = unique_targets[unique_targets['coverage_count'] > 0]
covered_weight = covered_targets['weight'].sum()
weighted_coverage_degree = covered_weight / total_weight if total_weight != 0 else 0

# (2) Weighted Average Coverage Count (WACC)
#     = (∑_{t in covered} (coverage_count(t) * w(t))) / (∑_{t in covered} w(t))
weighted_sum = (covered_targets['coverage_count'] * covered_targets['weight']).sum()
weighted_avg_coverage = weighted_sum / covered_weight if covered_weight != 0 else 0

# (3) Coverage Duplication (CDup): For targets with coverage_count > 1, average coverage count.
dup_targets = unique_targets[unique_targets['coverage_count'] > 1]
coverage_duplication = dup_targets['coverage_count'].mean() if len(dup_targets) > 0 else 0

# (4) Standard Deviation of Coverage Duplication (σ)
std_coverage_duplication = dup_targets['coverage_count'].std() if len(dup_targets) > 0 else 0

# (5) Coverage Orientation Dispersion (COD)
#     For targets with coverage by more than one camera, compute the maximum pairwise horizontal (yaw) angle difference, then take the average.
orientation_dispersion_list = []
penalties = []
for target_id, yaw_list in target_cameras.items():
    if len(yaw_list) > 1:
        # Compute all pairwise differences (in radians), choose the maximum and convert to degrees.
        max_diff = 0
        for a, b in itertools.combinations(yaw_list, 2):
            diff = abs(a - b)
            diff = min(diff, 2 * math.pi - diff)  # ensure the minimal difference on circle
            diff_deg = math.degrees(diff)
            if diff_deg > max_diff:
                max_diff = diff_deg
        if max_diff < 40:
            penalty = 40 - max_diff
        elif max_diff > 150:
            penalty = max_diff - 150
        else:
            penalty = 0
        penalties.append(penalty)
        orientation_dispersion_list.append(max_diff)
if len(orientation_dispersion_list) > 0:
    coverage_orientation_dispersion = np.mean(orientation_dispersion_list)
else:
    coverage_orientation_dispersion = 0

# (6) Coverage Orientation Dispersion Penalty (CODP)
#     If the average maximum pairwise horizontal angle (in degrees) deviates from an ideal value,
#     penalty = linearly increasing with deviation.

if penalties:
    coverage_orientation_dispersion_penalty = np.mean(penalties)
else:
    coverage_orientation_dispersion_penalty = 0


# ----------------------------
# 输出结果
# ----------------------------
print("被覆盖的目标点：", len(coverage_count))
print("Weighted Coverage Degree (WCD):", weighted_coverage_degree)
print("Weighted Average Coverage Count (WACC):", weighted_avg_coverage)
print("Coverage Duplication (CDup):", coverage_duplication)
print("Standard Deviation of Coverage Duplication (σ):", std_coverage_duplication)
print("Coverage Orientation Dispersion (COD, in degrees):", coverage_orientation_dispersion)
print("Coverage Orientation Dispersion Penalty (CODP):", coverage_orientation_dispersion_penalty)
# ----------------------------
# 可视化：绘制目标点覆盖情况散点图
# ----------------------------
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(unique_targets['Xt'], unique_targets['Yt'],
#                       c=unique_targets['coverage_count'], cmap='viridis', s=100, edgecolors='k')
# plt.xlabel('Xt')
# plt.ylabel('Yt')
# plt.title('Target Coverage Count Map')
# cbar = plt.colorbar(scatter)
# cbar.set_label('Coverage Count')
# plt.grid(True)
# plt.show()
# 根据 coverage_count 升序排序：使得覆盖次数较小的点先绘制，较大值的点后绘制，从而显示在上层
unique_targets.to_csv('E:\\camera\\lzy\\respoint2500.csv')
unique_targets.loc[unique_targets['coverage_count'] > 6, 'coverage_count'] -= 2
sorted_targets = unique_targets.sort_values(by='coverage_count', ascending=True)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# plt.figure(figsize=(8, 6))
#
# # 绘制“光晕”效果：用较大的 marker 和较低的透明度绘制底层散点图
# halo = plt.scatter(sorted_targets['Xt'], sorted_targets['Yt'],
#                    c=sorted_targets['coverage_count'], cmap='viridis',
#                    s=100, alpha=0.3, edgecolors='none', zorder=5)
#
# # 绘制真实的散点
# points = plt.scatter(sorted_targets['Xt'], sorted_targets['Yt'],
#                      c=sorted_targets['coverage_count'], cmap='viridis',
#                      s=20, edgecolors='k', zorder=10)
#
# plt.xlabel('Xt')
# plt.ylabel('Yt')
# plt.title('覆盖散点图')
# cbar = plt.colorbar(points)
# cbar.set_label('覆盖次数')
# plt.grid(True)
# plt.show()
# 获取 viridis 色图的 256 个颜色
# 创建自定义颜色映射：使得值为0时为黑色，值为1时为黄色，然后逐渐变为橙色
# 根据 coverage_count 升序排序：覆盖次数较小的先绘制，覆盖次数较大的后绘制
sorted_targets = unique_targets.sort_values(by='coverage_count', ascending=True)

# 将 coverage_count 列转换为浮点数数组
c_values = sorted_targets['coverage_count'].astype(float).values

# 定义自定义 colormap：当覆盖次数为 0 时显示黑色，覆盖次数大于0时从黄色开始过渡到橙色
colors = [(0, 0, 0), (1, 1, 0), (1, 0.5, 0)]
custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 设置归一化范围，确保 0 映射为黑色
norm = mpl.colors.Normalize(vmin=0, vmax=c_values.max())

plt.figure(figsize=(8, 6))
scatter = plt.scatter(sorted_targets['Xt'], sorted_targets['Yt'],
                      c=c_values, cmap=custom_cmap, norm=norm,
                      s=20, edgecolors='k', zorder=10)

# plt.xlabel("Xt")
# plt.ylabel("Yt")
plt.title("Coverage Scatterplot")
cbar = plt.colorbar(scatter)
cbar.set_label("Number of times covered")
plt.grid(True)
plt.show()

# 使用 np.histogram2d 聚合数据为二维直方图
bins = 70  # 网格数量，可以根据数据量调整
heatmap, xedges, yedges = np.histogram2d(sorted_targets['Xt'], sorted_targets['Yt'],
                                          bins=bins, weights=sorted_targets['coverage_count'])

custom_cmap = mpl.colors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

# 设置归一化范围
norm = mpl.colors.Normalize(vmin=0, vmax=np.max(heatmap))

# 绘制热力图
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, origin='lower', cmap=custom_cmap, norm=norm,
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
plt.colorbar(label='degree of coverage')
# plt.xlabel('Xt')
# plt.ylabel('Yt')
plt.title('Heat map of the number of times covered')
plt.grid(True)
plt.show()