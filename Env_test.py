import copy

import torch
import numpy as np
from norm import MeanStdNormalizer,MinMaxNormalizer,LogNormalizer,DynamicNormalizer
class nd_ris(object):

    def __init__(self,N,M):
        # 直接定义系统固定参数
        self.N = N # RIS元素阵元数
        self.M = M # 基站天线数
        self.K = 4  # 用户数量
        self.B = 100  # 带宽
        self.pho = 0.01
        self.kappa = 6  # 用户与RIS的莱斯因子
        self.kappa_rb = 12  # ris与基站的莱斯因子
        self.alpha_kb = 3.5  # 基站与用户直连路径损失指数
        self.alpha_kr = 2.5  # 用户与ris路径损失指数
        self.alpha_rb = 2  # 基站与ris路径损失指数
        self.noise = 1e-12
        self.posA = np.array([35.0, 150.0, 20.0]) #bs位置
        self.kappa_kb = 3  # YONGHU 与基站的莱斯因子
        self.posB = np.array([30.0, 145.0, 15.0]) # RIS位置
        self.pose = np.array([36.0, 125.0, 2.0])#窃听者位置
        self.kappa_eb = 4
        self.kappa_er = 8
        self.alpha_eb = 3.2
        self.alpha_er = 2.5
        # 用户位置
        self.pos1 = np.array([37.0, 122.0, 2.0])
        self.pos2 = np.array([39.0, 124.0, 2.0])
        self.pos3 = np.array([41.0, 126.0, 2.0])
        self.pos4 = np.array([43.0, 128.0, 2.0])

        self.posU = [self.pos1, self.pos2, self.pos3, self.pos4]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.out_secure = 0
        self.out_random_bf = 0
        self.out_NOcrack_mrt = 0
        self.out_NOcrack_zf = 0
        self.out_crack_mrt = 0
        self.out_crack_zf = 0
        self.out_random_mrt = 0
        self.out_random_zf = 0

    def reset(self):
        self.done = False
        state =self.get_state()


        return state

    def get_state(self):
        """返回状态值，状态值的数据类型为np.array"""
        self.Hur, self.Hub, self.Hrb,self.HubL = self.generate_channel()

        #disco ris
        # matrix = np.random.randn(self.N, 1)
        # complex_matrix = np.exp(2j * np.pi * matrix)
        # diagonal_matrix = np.diag(np.squeeze(complex_matrix))
        # identity_matrix = np.eye(self.N, dtype=np.complex128)
        # nr_ris = diagonal_matrix - identity_matrix
        # self.uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris)

        self.uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb,self.nr_ris)

        # 提取实部和虚部
        real_part = np.real(self.uplink)
        imag_part = np.imag(self.uplink)
        # 将实部和虚部放在一起
        # 初始化归一化器
        real_normalizer = MeanStdNormalizer()
        imag_normalizer = MeanStdNormalizer()
        # real_normalizer = LogNormalizer()
        # imag_normalizer = LogNormalizer()


        # 假设你有历史数据用于拟合归一化器

        real_normalizer.fit(real_part)  # 拟合实部
        imag_normalizer.fit(imag_part)  # 拟合虚部

        # 对数归一化实部和虚部
        real_part_norm = real_normalizer.transform(real_part)
        imag_part_norm = imag_normalizer.transform(imag_part)

        # 将对数归一化后的实部和虚部堆叠在一起
        combined = np.stack((real_part_norm, imag_part_norm))
        # combined = np.stack((real_part, imag_part))
        # 将结果转换为一维列表
        # state = combined.flatten().tolist()

        return combined

    def step(self,action,step):
        step += 1

        H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb,self.nr_ris)
        reward = self.compute_reward(H_down_real,self.action_w_normal(action))
        # reward_mrt = self.compute_reward_attack(H_down_real, self.uplink.T)
        #
        # reward_direct = self.compute_reward_attack(self.Hub.T,self.Hub.T)
        reward_mrt = 0
        reward_direct = 0
        print("reward",reward)
        print("reward_mrt",reward_mrt)
        print("reward_direct",reward_direct)


        state_ = self.get_state()

        if step == 50:
            self.done = True
        reward_normalizer = DynamicNormalizer(window_size=100)
        reward=reward_normalizer.normalize(reward)
        return state_, reward, reward_mrt,reward_direct,self.done

    def step_test(self,action,step):
        step += 1
        self.H_er, self.H_eb = self.generate_eve_channel()
        H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb,self.nr_ris)
        H_down_real_eve = self.downlink_channel_compute(self.H_er, self.H_eb, self.Hrb, self.nr_ris)
        reward,rate_secure = self.compute_reward(H_down_real,self.action_w_normal(action))

        reward_eve_secure,rate_eve_secure = self.compute_reward_eve(H_down_real_eve,self.action_w_normal(action) )
        reward_eve_random_bf, rate_eve_random_bf = self.compute_reward_eve(H_down_real_eve, self.action_w_random())
        reward_eve_crack_mrt, rate_eve_crack_mrt = self.compute_reward_eve(H_down_real_eve, self.MRT_precoding(self.uplink.T))
        reward_eve_crack_zf, rate_eve_crack_zf = self.compute_reward_eve(H_down_real_eve,
                                                                           self.ZF_precoding(self.uplink.T))
        reward_secrecy = reward-reward_eve_secure

        reward_random_bf,rate_random_bf = self.compute_reward(H_down_real,self.action_w_random())

        reward_mrt,rate_mrt = self.compute_reward_attack(H_down_real, self.uplink.T)
        reward_zf,rate_zf = self.compute_reward_attack_zf(H_down_real, self.uplink.T)
        # reward_zf =0

        reward_direct_mrt,rate_direct_mrt = self.compute_reward_attack(self.Hub.T,self.Hub.T)
        reward_direct_zf,rate_direct_zf = self.compute_reward_attack_zf(self.Hub.T, self.Hub.T)

        reward_secure =0
        reward_random_bf_secure =0
        reward_crack_mrt_secure =0
        reward_crack_zf_secure =0
        for i in range(self.K):
            reward_secure += rate_secure[i] - rate_eve_secure[i]
            reward_random_bf_secure += max((rate_random_bf[i] - rate_eve_random_bf[i]),0)
            reward_crack_mrt_secure += max((rate_mrt[i] - rate_eve_crack_mrt[i]), 0)
            reward_crack_zf_secure += max((rate_zf[i] - rate_eve_crack_zf[i]), 0)


        pair_order = np.random.permutation(self.N)
        nr_ris = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N // 2):
            theta1 = np.exp(2j * np.pi * np.random.randn())
            nr_ris[pair_order[2 * i], pair_order[2 * i - 1]] = theta1
            nr_ris[pair_order[2 * i - 1], pair_order[2 * i]] = -theta1

        uplink = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb,nr_ris)
        H_down_real_random_ris = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb,nr_ris)
        reward_random_mrt,rate_random_mrt = self.compute_reward_attack(H_down_real_random_ris,uplink.T)
        reward_random_zf,rate_random_zf = self.compute_reward_attack_zf(H_down_real_random_ris, uplink.T)

        state_ = None
        if step == 50:
            self.done = True

        for i in range(len(rate_secure)):
            if rate_secure[i] < rate_eve_secure[i]:
                self.out_secure+=1
            if rate_random_bf[i] < rate_eve_random_bf[i]:
                self.out_random_bf+=1
            # if rate_direct_mrt[i] < rate_eve[i]:
            #     self.out_NOcrack_mrt+=1
            # if rate_direct_zf[i] < rate_eve[i]:
            #     self.out_NOcrack_zf+=1
            if rate_mrt[i] < rate_eve_crack_mrt[i]:
                self.out_crack_mrt+=1
            if rate_zf[i] < rate_eve_crack_zf[i]:
                self.out_crack_zf+=1
            # if rate_random_mrt[i] < rate_eve[i]:
            #     self.out_random_mrt+=1
            # if rate_random_zf[i] < rate_eve[i]:
            #     self.out_random_zf+=1

        return state_,np.max(reward-reward_eve_secure,0),reward_secure,reward_random_bf_secure,reward_crack_mrt_secure,reward_crack_zf_secure,\
            reward, reward_random_bf,reward_random_mrt,reward_random_zf, reward_mrt,reward_direct_mrt,reward_zf,reward_direct_zf

    def step_test_disco(self, action, step):
        step += 1


        reward=0
        reward_random_bf = 0
        reward_zf = 0
        reward_mrt = 0
        reward_direct_mrt = 0
        reward_direct_zf = 0
        reward_random_mrt = 0
        reward_random_zf = 0

        reward_disco =0
        reward_eve_secure_disco =0
        reward_eve_random_bf_disco =0
        reward_eve_crack_mrt_disco =0
        reward_eve_crack_zf_disco =0
        reward_random_bf_disco =0
        reward_mrt_disco =0
        reward_zf_disco =0
        reward_random_mrt_disco =0
        reward_random_zf_disco =0
        reward_direct_mrt_disco =0
        reward_direct_zf_disco =0

        reward_secure_disco = 0
        reward_random_bf_secure_disco = 0
        reward_crack_mrt_secure_disco = 0
        reward_crack_zf_secure_disco = 0

        for i in range(1):
            matrix = np.random.randn(self.N, 1)
            complex_matrix = np.exp(2j*np.pi*matrix)
            diagonal_matrix = np.diag(np.squeeze(complex_matrix))
            identity_matrix = np.eye(self.N, dtype=np.complex128)
            nr_ris = diagonal_matrix - identity_matrix


            self.H_er, self.H_eb = self.generate_eve_channel()
            H_down_real = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris)
            H_down_real_eve = self.downlink_channel_compute(self.H_er, self.H_eb, self.Hrb, nr_ris)
            reward, rate_secure = self.compute_reward(H_down_real, self.action_w_normal(action))

            reward_eve_secure, rate_eve_secure = self.compute_reward_eve(H_down_real_eve, self.action_w_normal(action))
            reward_eve_random_bf, rate_eve_random_bf = self.compute_reward_eve(H_down_real_eve, self.action_w_random())
            reward_eve_crack_mrt, rate_eve_crack_mrt = self.compute_reward_eve(H_down_real_eve,
                                                                               self.MRT_precoding(self.uplink.T))
            reward_eve_crack_zf, rate_eve_crack_zf = self.compute_reward_eve(H_down_real_eve,
                                                                             self.ZF_precoding(self.uplink.T))


            reward_random_bf, rate_random_bf = self.compute_reward(H_down_real, self.action_w_random())

            reward_mrt, rate_mrt = self.compute_reward_attack(H_down_real, self.uplink.T)
            reward_zf, rate_zf = self.compute_reward_attack_zf(H_down_real, self.uplink.T)
            # reward_zf =0

            reward_direct_mrt, rate_direct_mrt = self.compute_reward_attack(self.Hub.T, self.Hub.T)
            reward_direct_zf, rate_direct_zf = self.compute_reward_attack_zf(self.Hub.T, self.Hub.T)



            reward_disco += reward
            reward_eve_secure_disco += reward_eve_secure
            reward_eve_random_bf_disco += reward_eve_random_bf
            reward_eve_crack_mrt_disco += reward_eve_crack_mrt
            reward_eve_crack_zf_disco += reward_eve_crack_zf
            reward_random_bf_disco += reward_random_bf
            reward_mrt_disco += reward_mrt
            reward_zf_disco += reward_zf
            reward_direct_mrt_disco += reward_direct_mrt
            reward_direct_zf_disco += reward_direct_zf

            # reward_secrecy_disco = reward_disco - reward_eve_secure_disco

            for i in range(self.K):
                reward_secure_disco += rate_secure[i] - rate_eve_secure[i]
                reward_random_bf_secure_disco += max((rate_random_bf[i] - rate_eve_random_bf[i]), 0)
                reward_crack_mrt_secure_disco += max((rate_mrt[i] - rate_eve_crack_mrt[i]), 0)
                reward_crack_zf_secure_disco += max((rate_zf[i] - rate_eve_crack_zf[i]), 0)

            matrix = np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)
            identity_matrix = np.eye(self.N, dtype=np.complex128)
            nr_ris_random = matrix - identity_matrix



            uplink_random = self.uplink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris_random)
            H_down_real_random_ris = self.downlink_channel_compute(self.Hur, self.Hub, self.Hrb, nr_ris_random)
            reward_random_mrt , rate_random_mrt= self.compute_reward_attack(H_down_real_random_ris,uplink_random.T)
            reward_random_zf , rate_random_zf= self.compute_reward_attack_zf(H_down_real_random_ris, uplink_random.T)

            reward_random_mrt_disco += reward_random_mrt
            reward_random_zf_disco += reward_random_zf

        state_ = None
        if step == 50:
            self.done = True
        return state_, reward_disco - reward_eve_crack_mrt_disco, reward_secure_disco,reward_random_bf_secure_disco,reward_crack_mrt_secure_disco, reward_crack_zf_secure_disco,\
            reward_disco, reward_random_bf_disco, \
            reward_random_mrt_disco, reward_random_zf_disco, reward_mrt_disco, reward_direct_mrt_disco, reward_zf_disco, reward_direct_zf_disco


    def get_NR_array(self,Hur_all, Hub_all, Hrb_all,malicious):
        if malicious:
            t = 100
        else:
            t=1

        rate = 0
        list = []
        for j in range(t):
            pair_order = np.random.permutation(self.N)
            nr_ris = np.zeros((self.N, self.N), dtype=complex)
            for i in range(self.N // 2):
                theta1 = np.exp(2j * np.pi * np.random.randn())
                nr_ris[pair_order[2 * i], pair_order[2 * i - 1]] = theta1
                nr_ris[pair_order[2 * i - 1], pair_order[2 * i]] = -theta1

            for i in range(t):
                Hur = Hur_all[i]
                Hub = Hub_all[i]
                Hrb = Hrb_all[i]
                reward_direct_mrt, rate_direct_mrt= self.compute_reward_attack(Hub.T, Hub.T)
                reward_direct_zf,rate_direct_zf = self.compute_reward_attack_zf(Hub.T, Hub.T)
                uplink = self.uplink_channel_compute(Hur, Hub, Hrb, nr_ris)
                H_down = uplink.T
                H_down_real = self.downlink_channel_compute(Hur, Hub, Hrb, nr_ris)
                reward_mrt,rate_mrt = self.compute_reward_attack(H_down_real, H_down)
                reward_zf,rate_zf =self.compute_reward_attack_zf(H_down_real, H_down)
                rate += ((reward_mrt / reward_direct_mrt) +
                         (reward_zf / reward_direct_zf))

                # rate += (self.compute_reward_attack_zf(H_down_real, H_down))
                # rate+= (self.compute_reward_attack(H_down_real, H_down))
            list.append(rate)
            if j == 0:
                rate_low = rate
                self.nr_ris = nr_ris
            else:
                if rate < rate_low:
                    self.nr_ris = nr_ris
                    rate_low = rate
            rate = 0

    def distance(self, x, y):
        """计算x与y之间的距离，输出单位为米"""
        dis = np.linalg.norm(x - y)
        return dis

    def angle(self, x, y):
        """计算x与y之间的夹角，输出单位为弧度"""
        numerator = np.sqrt((y[0] - x[0]) ** 2 )
        denominator = np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 )
        ang = np.arccos(numerator / denominator)
        return ang

    def user_ris_channel(self, AoA_kr, dkr):
        """生成单个用户和ris之间的信道，输出为实部和虚部，维度为（N,1）"""
        indices = np.arange(self.N).reshape(-1, 1)
        hkrN = np.random.normal(0, np.sqrt(0.5), (self.N, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, 1))
        hkrL = np.exp(-1j * np.pi * indices * np.cos(AoA_kr))
        h_kr = np.sqrt(self.pho * (dkr ** (-self.alpha_kr))) * (
                (np.sqrt(self.kappa / (1 + self.kappa)) * hkrL) + (np.sqrt(1 / (1 + self.kappa)) * hkrN))

        return np.real(h_kr), np.imag(h_kr)

    def user_base_channel(self,AoA_kb, dkb):
        """生成单个用户和基站之间的信道，输出为实部和虚部，维度为（N,1）"""
        indices = np.arange(self.M).reshape(-1, 1)
        hubN = np.random.normal(0, np.sqrt(0.5), (self.M, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, 1))
        hubL = np.exp(-1j * np.pi * indices * np.cos(AoA_kb))
        h_ub = np.sqrt(self.pho * (dkb ** (-self.alpha_kb))) * (
                (np.sqrt(self.kappa_kb / (1 + self.kappa_kb)) * hubL) + (np.sqrt(1 / (1 + self.kappa_kb)) * hubN))
        return np.real(h_ub), np.imag(h_ub),np.real(hubL),np.imag(hubL)

    def ris_base_channel(self, drb, AoA_rb, AoD_rb):
        """生成ris和基站之间的信道，输出为实部和虚部，维度为（M,N）"""
        indices = np.arange(self.N).reshape(-1, 1)
        indices2 = np.arange(self.M).reshape(-1, 1)
        H_rbN = np.random.normal(0, np.sqrt(0.5), (self.M, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                            (self.M, self.N))

        H_rbL = (np.exp(-1j * np.pi * indices2 * np.cos(AoA_rb))) @ ((np.exp(-1j * np.pi * indices * np.cos(AoD_rb))).conjugate().T)


        H_rb = np.sqrt(self.pho * (drb ** (-self.alpha_rb))) * (
                (np.sqrt(self.kappa_rb / (1 + self.kappa_rb)) * H_rbL) + (np.sqrt(1 / (1 + self.kappa_rb)) * H_rbN))

        return H_rb

    def generate_eve_channel(self):
        der = self.distance(self.posB, self.pose)
        AoA_kr = self.angle(self.pose, self.posB)
        her_real, her_imag = self.eve_ris_channel(AoA_kr, der)
        Her = her_real + 1j * her_imag

        deb = self.distance(self.posA, self.pose)
        AoA_eb = self.angle(self.pose, self.posA)
        heb_real, heb_imag = self.eve_base_channel(AoA_eb,deb)
        Heb = heb_real + 1j * heb_imag



        return Her, Heb

    def eve_ris_channel(self, AoA_kr, dkr):
        """生成单个用户和ris之间的信道，输出为实部和虚部，维度为（N,1）"""
        indices = np.arange(self.N).reshape(-1, 1)
        hkrN = np.random.normal(0, np.sqrt(0.5), (self.N, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.N, 1))
        hkrL = np.exp(-1j * np.pi * indices * np.cos(AoA_kr))
        h_kr = np.sqrt(self.pho * (dkr ** (-self.alpha_er))) * (
                (np.sqrt(self.kappa_er / (1 + self.kappa_er)) * hkrL) + (np.sqrt(1 / (1 + self.kappa_er)) * hkrN))

        return np.real(h_kr), np.imag(h_kr)

    def eve_base_channel(self, AoA_kb, dkb):
        """生成单个用户和基站之间的信道，输出为实部和虚部，维度为（N,1）"""
        indices = np.arange(self.M).reshape(-1, 1)
        hubN = np.random.normal(0, np.sqrt(0.5), (self.M, 1)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.M, 1))
        hubL = np.exp(-1j * np.pi * indices * np.cos(AoA_kb))
        h_ub = np.sqrt(self.pho * (dkb ** (-self.alpha_eb))) * (
                (np.sqrt(self.kappa_eb / (1 + self.kappa_eb)) * hubL) + (np.sqrt(1 / (1 + self.kappa_eb)) * hubN))
        return np.real(h_ub), np.imag(h_ub)

    def generate_channel(self):
        """生成ris-base,user-ris,user-base之间的信道，均为复数矩阵"""
        Hur_real = np.empty([self.N, self.K])
        Hur_imag = np.empty([self.N, self.K])
        Hub_real = np.empty([self.M, self.K])
        Hub_imag = np.empty([self.M, self.K])

        HubL_real = np.empty([self.M, self.K])
        HubL_imag = np.empty([self.M, self.K])

        drb = self.distance(self.posA, self.posB)
        AoA_rb = self.angle(self.posB, self.posA)
        AoD_rb = AoA_rb

        Hrb = self.ris_base_channel(drb, AoA_rb, AoD_rb)

        for i in range(self.K):
            pos = self.posU[i]
            dkr = self.distance(self.posB, pos)
            dkb = self.distance(self.posA, pos)
            AoA_kr = self.angle(pos, self.posB)
            AoA_kb = self.angle(pos, self.posA)
            hkr_real, hkr_imag = self.user_ris_channel(AoA_kr, dkr)
            hkb_real, hkb_imag,hkbL_real,hkbL_imag = self.user_base_channel(AoA_kb, dkb)

            Hur_real[:, i] = hkr_real.squeeze()
            Hur_imag[:, i] = hkr_imag.squeeze()
            Hub_real[:, i] = hkb_real.squeeze()
            Hub_imag[:, i] = hkb_imag.squeeze()

            HubL_real[:, i] = hkbL_real.squeeze()
            HubL_imag[:, i] = hkbL_imag.squeeze()
        Hur = Hur_real + 1j * Hur_imag
        Hub = Hub_real + 1j * Hub_imag
        HubL = HubL_real + 1j * HubL_imag
        return Hur, Hub, Hrb,HubL
    def uplink_channel_compute(self, Hur, Hub, Hrb,nr_ris):

        uplink = (Hrb @ nr_ris @ Hur) + Hub
        return uplink

    def downlink_channel_compute(self, Hur, Hub, Hrb,nr_ris):
        """计算实际的下行信道"""

        downlink = (Hur.T @ nr_ris @ Hrb.T) + Hub.T
        return downlink


    def compute_reward(self, H_real_down, action):
        """该函数用于计算奖励值，奖励值设定为用户和速率的倒数，速率越低奖励值越高"""
        W = action
        # print('action',action)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k]))** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = (np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2))
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
            print('rate',k,rate[k])
        sum_rate = np.sum(rate)
        reward = sum_rate
        return reward,rate

    def compute_reward_eve(self, H_real_down_eve, action):
        """该函数用于计算奖励值，奖励值设定为用户和速率的倒数，速率越低奖励值越高"""
        W = action
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down_eve, W[:, k]))** 2)
            W_removed = np.delete(W, k, axis=1)
            interference = (np.sum(np.abs(np.dot(H_real_down_eve, W_removed)) ** 2))
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
            print('eve',k,rate[k])
        sum_rate = np.sum(rate)
        reward = sum_rate
        return reward,rate

    def compute_reward_attack(self, H_real_down, H_down):
        """该函数用于计算奖励值，奖励值设定为用户和速率的倒数，速率越低奖励值越高"""
        W = self.MRT_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = (np.abs(np.dot(H_real_down[k, :], W[:, k]))** 2)
            W_removed = np.delete(W, k, axis=1)
            interference =(np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2))
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        sum_rate = np.sum(rate)
        reward = sum_rate
        return reward,rate


    def compute_reward_attack_zf(self, H_real_down, H_down):
        """该函数用于计算奖励值，奖励值设定为用户和速率的倒数，速率越低奖励值越高"""
        W = self.ZF_precoding(H_down)
        rate = np.zeros(self.K)
        for k in range(self.K):
            sig = np.abs(np.dot(H_real_down[k, :], W[:, k]))** 2
            W_removed = np.delete(W, k, axis=1)
            interference = np.sum(np.abs(np.dot(H_real_down[k, :], W_removed)) ** 2)
            sinr = sig / (interference + self.noise)
            rate[k] = self.B * np.log2(1 + sinr)
        sum_rate = np.sum(rate)
        reward = sum_rate
        return reward,rate

    def ZF_precoding(self, H):
        """该函数用于实现ZF预编码，输出的为归一化后的ZF预编码矩阵"""
        H_hermitian = np.conj(H).T
        W_zf = np.linalg.pinv(H_hermitian @ H) @ H_hermitian
        W_zf_normalized = W_zf / np.linalg.norm(W_zf, 'fro')
        return W_zf_normalized

    def MRT_precoding(self, H):
        """该函数用于实现ZF预编码，输出的为归一化后的ZF预编码矩阵"""
        H_hermitian = np.conj(H).T
        W_MRT = H_hermitian
        frobenius_norm = np.linalg.norm(W_MRT, 'fro')
        W_MRT_normalized = W_MRT / frobenius_norm

        return  W_MRT_normalized

    def action_w(self,action):
        action_w = np.empty([self.M, self.K], dtype=complex)  # 指定数据类型为复数
        for i in range(self.K):
            real_part = action[self.M * i:self.M * (i + 1)]
            imag_part = action[self.M * (i + 1):self.M * (i + 2)]
            # 循环遍历实部和虚部对应的元素列表来构建每列的复数元素
            for n in range(self.M):
                action_w[n, i] = complex(real_part[n], imag_part[n])
        frobenius_norm = np.linalg.norm(action_w, 'fro')
        action_w_normalized = action_w / frobenius_norm
        return action_w_normalized

    def action_w_normal(self, action):
         # 将后N * K个元素的值变为0 - 2 * np.pi之间（利用向量化操作）
        w = copy.deepcopy(action)
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi
        # 使用向量化操作将二维数组中的每个元素对应的角度值转换为复数（利用cos和sin函数的向量化计算）
        action_w = w.reshape((2, self.M, self.K))
        # 根据幅度和相位构建复数二维矩阵（利用向量化操作）
        action_w = np.vectorize(lambda amplitude, phase: amplitude * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        frobenius_norm = np.linalg.norm(action_w, 'fro')
        action_w_normalized = action_w / frobenius_norm
        print("action_power",np.linalg.norm(action_w_normalized))


        return action_w

    def action_w_random(self):
         # 将后N * K个元素的值变为0 - 2 * np.pi之间（利用向量化操作）
        w =np.array(np.random.randn(2*self.M*self.K).tolist())
        w[self.M * self.K:] = w[self.M * self.K:] * 2 * np.pi
        # 使用向量化操作将二维数组中的每个元素对应的角度值转换为复数（利用cos和sin函数的向量化计算）
        action_w = w.reshape((2, self.M, self.K))
        # 根据幅度和相位构建复数二维矩阵（利用向量化操作）
        action_w = np.vectorize(lambda amplitude, phase: amplitude * (np.cos(phase) + 1j * np.sin(phase)))(
            action_w[0], action_w[1])
        frobenius_norm = np.linalg.norm(action_w, 'fro')
        action_w_normalized = action_w / frobenius_norm
        print("action_power",np.linalg.norm(action_w_normalized))
        return action_w