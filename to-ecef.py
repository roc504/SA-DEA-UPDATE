import pandas as pd
import math
import numpy as np

# 定义经纬度和高程转换为 ECEF 坐标的函数
def geodetic_to_ecef(lon, lat, alt):
    # 将角度从度转换为弧度
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    # WGS84 参数
    a = 6378137.0               # 长半轴（单位：米）
    f = 1 / 298.257223563       # 扁率
    b = a * (1 - f)             # 短半轴
    e_sq = 1 - (b**2 / a**2)      # 第一偏心率的平方
    # 计算卯酉圈曲率半径
    N = a / math.sqrt(1 - e_sq * (math.sin(lat_rad) ** 2))
    # 计算 ECEF 坐标
    X = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - e_sq) * N + alt) * math.sin(lat_rad)
    return X, Y, Z

# 读取融合后的 CSV 文件
df = pd.read_csv('E:\\camera\\data\\map.csv')

# 针对 test 数据，假设经纬高分别存放在 'Xt', 'Yt', 'Zt'
def compute_test_ecef(row):
    if pd.isna(row['lon']) or pd.isna(row['lat']) or pd.isna(row['H']):
        # 指定返回 Series 的索引，保证长度一致
        return pd.Series([None, None, None], index=['Xt_ecef', 'Yt_ecef', 'Zt_ecef'])
    return pd.Series(geodetic_to_ecef(row['lon'], row['lat'], row['H']),
                     index=['X', 'Y', 'Z'])

# 新增 test 数据对应的 ECEF 坐标列
df[['X', 'Y', 'Z']] = df.apply(compute_test_ecef, axis=1)
#
# # 针对 can 数据，假设经纬高分别存放在 'Xc', 'Yc', 'Zc'
# def compute_can_ecef(row):
#     if pd.isna(row['lon_c']) or pd.isna(row['lat_c']) or pd.isna(row['Hc']):
#         return pd.Series([None, None, None], index=['Xc_ecef', 'Yc_ecef', 'Zc_ecef'])
#     return pd.Series(geodetic_to_ecef(row['lon_c'], row['lat_c'], row['Hc']),
#                      index=['Xc_ecef', 'Yc_ecef', 'Zc_ecef'])
#
# # 新增 can 数据对应的 ECEF 坐标列
# df[['Xc_ecef', 'Yc_ecef', 'Zc_ecef']] = df.apply(compute_can_ecef, axis=1)
# # 计算两点之间的欧几里得距离（单位：米）
# df['distance_m'] = np.sqrt((df['Xt_ecef'] - df['Xc_ecef'])**2 +
#                            (df['Yt_ecef'] - df['Yc_ecef'])**2 +
#                            (df['Zt_ecef'] - df['Zc_ecef'])**2)
# df_filtered = df[df['distance_m'] <= 100]
# # 保存包含距离的新 CSV 文件
# df_filtered.to_csv('E:\\camera\\df_merged_with_distance100.csv', index=False)
# # 保存转换后的结果到新的 CSV 文件
df.to_csv('E:\\camera\\map_ecef.csv', index=False)

# 输出前几行查看结果
print(df.head())
