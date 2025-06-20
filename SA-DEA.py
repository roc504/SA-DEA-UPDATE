import numpy as np
import pandas as pd
import random
import time
import heapq
import math
import matplotlib.pyplot as plt
import itertools

def log2(n):
    return math.ceil(math.log2(n + 1))
class Population:
    def __init__(self, min_range, max_range, pre_pitch,pre_yaw,dim, rounds, size, object_func, alphaM=6, alphaC=6, bM=0.03,
                 bC=0.03, cCR=0.15, dCR=0.8,
                 alpha=0.2, cMR=0.6, dMR=0.2, beta=0.05):
        self.min_range = min_range
        self.max_range = max_range
        self.pre_pitch=pre_pitch
        self.pre_yaw=pre_yaw
        self.dimension = dim
        self.MR = 1
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.CR = 0.8
        self.alphaM = alphaM
        self.alphaC = alphaC
        self.bM = bM
        self.bC = bC
        self.cCR = cCR
        self.dCR = dCR
        self.alpha = alpha
        self.cMR = cMR
        self.dMR = dMR
        self.beta = beta
        self.tu=[]
        self.L=log2(self.max_range)+log2(180/self.pre_pitch)+log2(360/self.pre_yaw)
        # self.gama = gama
        self.cross = np.zeros((self.size, self.L * self.dimension))
        self.best = np.array([random.randint(0, 1) for s in range(self.L* self.dimension)])
        self.best2 = np.array([random.randint(0, 1) for s in range(self.L * self.dimension)])
        self.far = np.array([random.randint(0, 1) for s in range(self.L * self.dimension)])
        # np.array([random.randint(self.min_range, self.max_range) for s in range(self.dimension)])np.random.randint(2, size=(1, len(bin(max_range)[2:]) * self.dimension))
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = np.random.randint(2, size=(size, self.L * self.dimension))
        # self.individuality = [np.array([random.randint(self.min_range, self.max_range) for s in range(self.dimension)])
        #                       for tmp in range(size)]
        # result = self.individuality[1,:].tolist()
        # result1 = self.individuality[1, 7:14].tolist()
        # for x in self.individuality:
        #     xx=x[0:7]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            r = random.random()
            if r < 0.5 - (self.cur_round /self.size) *0.3:
                tmp = (1 - self.best) * abs(self.individuality[r1] - self.individuality[r2]) + (
                        self.individuality[r1] * self.individuality[r2])
            elif 0.5 -(self.cur_round /self.size) *0.3 <= r < 0.8 - (self.cur_round /self.size) *0.4:
                tmp = (1 - self.best2) * abs(self.individuality[r1] - self.individuality[r2]) + (
                        self.individuality[r1] * self.individuality[r2])
            else:
                tmp = (1 - self.far) * abs(self.individuality[r1] - self.individuality[r2]) + (
                        self.individuality[r1] * self.individuality[r2])
            # tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor
            # tmp = np.around(tmp)
            # tmp = tmp.astype(np.int8)
            # for t in range(self.dimension):
            #     if tmp[t] > self.max_range or tmp[t] < self.min_range:
            #         tmp[t] = random.randint(self.min_range, self.max_range)
            self.mutant.append(tmp)

    def crossover(self):
        # cross = np.zeros((self.size, len(bin(self.max_range)[2:]) * self.dimension))
        for i in range(self.size):
            MR1 = self.cMR * (1 / (1 + math.exp(self.alphaM - self.bM * self.cur_round))) + self.dMR
            CR1 = self.cCR * (1 / (1 + math.exp(self.alphaC - self.bC * self.cur_round))) + self.dCR
            favg = sum(self.object_function_values) / self.size
            fi = self.get_object_function_value(self.individuality[i])
            if fi >= favg:
                self.MR = MR1
                self.CR = CR1
            else:
                self.MR = MR1 - ((favg - fi) / favg) * self.alpha
                self.CR = CR1 - ((favg - fi) / favg) * self.beta
            # Jrand = random.randint(0, len(bin(self.max_range)[2:]) * self.dimension)
            for j in range(self.dimension):
                T = random.random()

                if T >= self.MR:
                    # for jj in range(len(bin(self.max_range)[2:])):
                    #     self.cross[i][j * len(bin(self.max_range)[2:]) + jj] = self.mutant[i][
                    #         j * len(bin(self.max_range)[2:]) + jj]
                    ran = random.randint(2, self.L - 1)
                    for jj in range(ran):
                        self.cross[i][j * self.L + jj] = self.individuality[i][
                            j * self.L + jj]
                    for jj in range(self.L - ran):
                        self.cross[i][j * self.L + ran + jj] = random.randint(0, 1)
                elif T < self.MR * (1 - self.CR):
                    for jj in range(self.L):
                        self.cross[i][j * self.L + jj] = self.individuality[i][
                            j * self.L+ jj]
                else:
                    # ran = random.randint(2, len(bin(self.max_range)[2:]) - 1)
                    # for jj in range(ran):
                    #     self.cross[i][j * len(bin(self.max_range)[2:]) + jj] = self.individuality[i][
                    #         j * len(bin(self.max_range)[2:]) + jj]
                    # for jj in range(len(bin(self.max_range)[2:]) - ran):
                    #     self.cross[i][j * len(bin(self.max_range)[2:]) + ran + jj] = random.randint(0, 1)
                    for jj in range(self.L):
                        self.cross[i][j * self.L+ jj] = self.mutant[i][
                            j * self.L + jj]
            # for j in range(len(bin(self.max_range)[2:]) * self.dimension):
            #     T = random.random()
            #     if T >= self.MR:
            #         self.cross[i][j] = random.randint(0, 1)
            #     elif T < self.MR * (1 - self.CR):
            #         self.cross[i][j] = self.individuality[i][j]
            #     else:
            #         # tt[0][j]tt =
            #         self.cross[i][j] = self.mutant[i][j]
            # decode(cross[i])
            # if random.random() > self.CR and j != Jrand:
            #     self.mutant[i][j] = self.individuality[i][j]

    def select(self):
        cro = []
        # indi = []
        temp = np.zeros((self.size, self.L * self.dimension))
        for i in range(self.size):
            cro.append(self.get_object_function_value(self.cross[i]))
            # indi.append(self.get_object_function_value(self.individuality[i]))
        cro.extend(self.object_function_values)
        max_number = heapq.nlargest(self.size, cro)
        max_index = []
        for t in max_number:
            index = cro.index(t)
            max_index.append(index)
        n = 0
        for m in range(len(max_index)):
            if max_index[m] >= self.size:
                temp[n] = self.individuality[max_index[m] - self.size]
                self.object_function_values[n] = max_number[m]
            else:
                temp[n] = self.cross[max_index[m]]
                self.object_function_values[n] = max_number[m]
            n = n + 1
        self.individuality = temp
        #
        # if tmp < self.object_function_values[i]:
        #     self.individuality[i] = self.cross[i]
        #     self.object_function_values[i] = tmp




    def print_best(self):
        max_number = heapq.nlargest(self.size, self.object_function_values)
        max_index = []
        for t in max_number:
            index = self.object_function_values.index(t)
            max_index.append(index)
        # m = max(self.object_function_values)
        # i = self.object_function_values.index(m)
        self.best = self.individuality[max_index[0]]
        self.best2 = self.individuality[max_index[1]]
        far=random.randint(int((self.size/3)*2),self.size-1)
        self.far = self.individuality[max_index[far]]
        self.tu.append(max_number[0])
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(decode(self.individuality[2])))
        print("目标函数值：" + str(max_number[0]))

    def evolution(self):
        while self.cur_round < self.rounds:
            # self.update_para()
            self.mutate()
            self.crossover()
            self.select()
            self.print_best()
            self.cur_round = self.cur_round + 1


# 测试部分
if __name__ == "__main__":
    start = time.process_time()

    min_range = 0
    max_range = 366
    dim = 25
    pre_pitch=1
    pre_yaw=1
    # MR = 0.7
    rounds = 2000
    size = 30
    fov_half_deg = 40
    fov_half_rad = math.radians(fov_half_deg)

    def decode(v):
        L = log2(max_range) + log2(180 / pre_pitch) + log2(360 / pre_yaw)
        k0=log2(max_range)
        k1=log2(180 / pre_pitch)
        k2=log2(360 / pre_yaw)
        de_DNA = []
        for i in range(dim):
            result = v[(i * L):(i * L) + L].tolist()

            res_point=result[:k0]
            res_pit = result[k0:k1+k0]
            res_yaw=result[k1+k0:]
            res_point1 = int(''.join(str(int(s)) for s in res_point),2)
            res_pit1 =  int(''.join(str(int(s)) for s in res_pit),2)
            res_yaw1 =  int(''.join(str(int(s)) for s in res_yaw),2)

            if  res_point1 > max_range:
                res_point1 = random.randint(min_range, max_range)
                while  res_point1 in de_DNA:
                    res_point1 = random.randint(min_range, max_range)
            if  res_pit1 > 180:
                res_pit1 = random.randint(0, 180)
            if  res_yaw1 > 180:
                res_yaw1 = random.randint(0, 360)
            de_DNA.append([res_point1,res_pit1,res_yaw1])
            #
            # de_DNA.append(random.randint(min_range, max_range))
        return de_DNA


    # def latlon_to_ecef(lat, lon, h):
    #     # WGS84 常数
    #     a = 6378137  # 地球半径（米）
    #     f = 1 / 298.257223563  # 扁率
    #     e2 = 2 * f - f ** 2  # 偏心率
    #     lat_r = math.radians(lat)
    #     lon_r = math.radians(lon)
    #
    #     # 计算地球曲率参数 N
    #     N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    #
    #     # 转换为笛卡尔坐标
    #     X = (N + h) * math.cos(lat_r) * math.cos(lon_r)
    #     Y = (N + h) * math.cos(lat_r) * math.sin(lon_r)
    #     Z = ((1 - e2) * N + h) * math.sin(lat_r)
    #
    #     return X, Y, Z


    # # 计算两个笛卡尔坐标点之间的欧几里得距离
    # def calculate_distance(x1, y1, z1, x2, y2, z2):
    #     return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


    # 判断地面点是否在摄像头的可视范围内
    # def is_point_in_camera_range(camera, point, max_distance=50, max_pitch=35, max_yaw=35):
    #     # 摄像头的经纬度和高程，俯仰角和偏航角
    #     lat_c, lon_c, h_c, pitch_angle, yaw_angle = camera
    #     # 地面点的经纬度和高程
    #     lat_p, lon_p, h_p = point
    #
    #     # 将摄像头和地面点转换为笛卡尔坐标
    #     x_c, y_c, z_c = latlon_to_ecef(lat_c, lon_c, h_c)
    #     x_p, y_p, z_p = latlon_to_ecef(lat_p, lon_p, h_p)
    #
    #     # 计算地面点与摄像头的距离
    #     distance = calculate_distance(x_c, y_c, z_c, x_p, y_p, z_p)
    #
    #     # 如果距离大于最大可视距离，返回 False
    #     if distance > max_distance:
    #         return False
    #
    #     # 计算俯仰角（pitch）
    #     pitch = math.degrees(math.atan2(z_p - z_c, math.sqrt((x_p - x_c) ** 2 + (y_p - y_c) ** 2)))
    #
    #     # 判断俯仰角是否在可视范围内
    #     if abs(pitch) > max_pitch:
    #         return False
    #
    #     # 计算偏航角（yaw）
    #     yaw = math.degrees(math.atan2(y_p - y_c, x_p - x_c))
    #
    #     # 判断偏航角是否在可视范围内
    #     if abs(yaw) > max_yaw:
    #         return False
    #
    #     # 如果都满足条件，返回 True
    #     return True



    def f(v2):
        visible_targets = set()
        cameras = decode(v2)
        # ----------------------------
        # 统计每个目标点被覆盖的次数
        df = pd.read_csv('E:\\camera\\data\\Visibility_Matrix.csv')
        # 记录每个目标点被覆盖的摄像头偏航角（用于计算视角分散度），注意：仅取水平（偏航角）部分
        target_cameras = {}  # key: OID_TARGET, value: list of camera yaw angles in radians
        coverage_count = {}  # 键为 OID_TARGET，值为覆盖次数

        # ----------------------------
        # 统计每个目标点被覆盖的次数
        for cam in cameras:
            cam_id, pitch_deg, yaw_deg = cam
            pitch_rad = math.radians(pitch_deg)
            yaw_rad = math.radians(yaw_deg)

            # 计算摄像头视向单位向量
            view_vector = np.array([
                math.cos(pitch_rad) * math.cos(yaw_rad),
                math.cos(pitch_rad) * math.sin(yaw_rad),
                math.sin(pitch_rad)
            ])

            # 根据摄像头编号筛选数据
            cam_df = df[df['OID_OBSERV'] == cam_id]

            # 用于存储该摄像头覆盖到的目标点ID，避免重复计数
            covered_targets_set = set()

            # 遍历该摄像头的每个候选记录
            for idx, row in cam_df.iterrows():
                cam_coords = np.array([row['Xc'], row['Yc'], row['Zc']])
                target_coords = np.array([row['Xt'], row['Yt'], row['Zt']])

                vec = target_coords - cam_coords
                norm_vec = np.linalg.norm(vec)
                if norm_vec == 0:
                    continue  # 避免除零错误
                vec_unit = vec / norm_vec

                # 计算摄像头视向与目标向量的夹角
                dot_val = np.dot(view_vector, vec_unit)
                dot_val = max(min(dot_val, 1.0), -1.0)
                angle = math.acos(dot_val)

                # 如果夹角小于视野半角，则认为目标点在视域内
                if angle <= fov_half_rad:
                    target_id = row['OID_TARGET']
                    covered_targets_set.add(target_id)

            # 对该摄像头覆盖到的每个目标点，仅计数一次
            for target_id in covered_targets_set:
                coverage_count[target_id] = coverage_count.get(target_id, 0) + 1

        # ----------------------------
        # 统计加权平均覆盖次数与加权覆盖度（仅针对被覆盖的目标点）
        # ----------------------------
        # 获取所有唯一地面目标点及其权重
        unique_targets = df[['OID_TARGET', 'weight']].drop_duplicates(subset='OID_TARGET')
        total_weight = unique_targets['weight'].sum()
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
        # # 针对被覆盖的目标点（coverage_count中存在的目标点）计算加权平均覆盖次数
        # covered_targets = unique_targets[unique_targets['OID_TARGET'].isin(coverage_count.keys())]
        # covered_total_weight = covered_targets['weight'].sum()
        #
        # # 计算加权平均覆盖次数：只考虑被覆盖的目标点
        # weighted_sum = 0.0  # 累加每个被覆盖目标点的覆盖次数乘以其权重
        # for idx, row in covered_targets.iterrows():
        #     target_id = row['OID_TARGET']
        #     weight = row['weight']
        #     count = coverage_count.get(target_id, 0)
        #     weighted_sum += count * weight
        #
        # weighted_avg_coverage = weighted_sum / covered_total_weight if covered_total_weight != 0 else 0
        # weighted_avg_coverage_gui=weighted_avg_coverage*0.5
        # # 计算加权覆盖度：被覆盖目标点的权重之和占所有目标点总权重之和的比例
        # weighted_coverage_degree = covered_total_weight / total_weight if total_weight != 0 else 0

        fitness=1#适应度函数公式
        # ----------------------------
        # 输出结果
        # ----------------------------
        # print("各目标点被覆盖的次数：", coverage_count)
        # print("总共有", len(coverage_count), "个地面目标点被至少一个摄像头覆盖")
        # print("加权平均覆盖次数（仅针对被覆盖的目标点）：", weighted_avg_coverage)
        # print("加权覆盖度：", weighted_coverage_degree)
        return fitness




    p = Population(min_range=min_range, max_range=max_range, pre_pitch=1,pre_yaw=1,dim=dim, rounds=rounds, size=size,
                   object_func=f)
    p.evolution()
    # 中间写上代码块
    end = time.process_time()
    print('BDE Running time: %s Seconds' % (end - start))
    l = []
    for i in range(rounds-1):
    # PATH为导出文件的路径和文件名
        l.append(i)
    data2 = pd.DataFrame(data=p.tu, index=None)
    data2.to_csv('E:\\camera\\lzy\\BDE_res0209.csv')
    plt.plot(l,p.tu)
    plt.show()


