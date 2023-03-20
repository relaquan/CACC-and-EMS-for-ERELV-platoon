# -*- coding: utf-8 -*-
"""
Extended range electric vehicle
"""

import pickle
# 实现任意对象与文本之间的相互转化、任意对象与二进制之间的相互转化、 Python 对象的存储及恢复。
import numpy as np
from scipy.interpolate import interp1d

class EREV(object):
    def __init__(self):
        self.R_wheel = 0.376        # 车轮半径
        self.RPM_2_rads = 2*np.pi/60   # rpm to rad/s
        self.Q_batt = 75 * 3600      # 电池容量 26.25 kWh = 75 * 350 / 1000
        self.G_f = 6.15
        map = pickle.load(open('s_map.pkl','rb'))  # s_mpa  
        self.Eng_eff = map['Eng_eff'] # 发动机map
        self.Mot_eff = map['Mot_eff'] # 驱动电机map
        self.Gen_eff = map['Gen_eff'] # 发电机map
        
        # 电池：SOC-开路电压
        SOC_list = [0,0.00132660038355303,0.0574708841412090,0.107025590196296,0.156580329178506,0.206134478930137,0.255687515404994,0.305240257178054,0.354795683551491,0.404476817165514,0.454031523456938,0.503585181935343,0.553137334622058,0.602688668725870,0.652241410558679,0.701796476729142,0.751350724430672,0.800904841303918,0.850591981005177,0.900297369762374,0.950130539134393,0.999900000000000,1]
        Batt_vol = [310.634972652632,310.634972652632,326.951157031579,329.458999831579,332.872264042105,335.715896273684,338.111014926316,340.582868147368,342.019372105263,343.171024421053,344.347621200000,345.680204905263,347.335128694737,349.668433263158,353.212415936842,356.773192105263,360.233421347368,363.941352947368,367.896574926316,372.075074905263,376.567860442105,381.712636042105,381.712636042105]
        self.V_oc = interp1d(SOC_list, Batt_vol, kind = 'linear', fill_value = 'extrapolate')    
        # SOC-充电电阻
        Resistance_of_charge = [0.278441000000000,0.278441000000000,0.278441000000000,0.220686000000000,0.219190000000000,0.205348000000000,0.191300000000000,0.187277000000000,0.178095000000000,0.172985000000000,0.169769000000000,0.169487000000000,0.174937000000000,0.185496000000000,0.208445000000000,0.213703000000000,0.206749000000000,0.198442000000000,0.191939000000000,0.189976000000000,0.214378000000000,0.296465000000000,0.296465000000000]
        self.R_chg = interp1d(SOC_list, Resistance_of_charge, kind = 'linear', fill_value = 'extrapolate')
        # SOC-放电电阻
        Resistance_of_discharge = [0.313944125557612,0.313944125557612,0.313944125557612,0.242065812972601,0.230789057570884,0.212882084646638,0.196703117398054,0.191970726117600,0.182268278911152,0.176602004689331,0.173003239603114,0.172594564949654,0.177777197273542,0.188139855710787,0.212047123247474,0.218402021824003,0.211830447697718,0.203822955457752,0.197028250778952,0.194459010902986,0.218377885152249,0.296468469580613,0.296468469580613]
        self.R_dis = interp1d(SOC_list, Resistance_of_discharge, kind = 'linear', fill_value = 'extrapolate')
        # 转矩限制
        self.Eng_t_max = map['Eng_t_max']
        self.Mot_t_max = map['Mot_t_max']
        self.Mot_t_min = map['Mot_t_min']
        self.Gen_t_max = map['Gen_t_max']
        self.find_te = map['find_te']
        self.find_we = map['find_we']

    def run(self, car_speed, car_torque, P, soc):
        
        car_speed = car_speed if car_speed > 0 else 0 
        
        # 驱动电机
        T_axle = car_torque / self.G_f   # 驱动电机转矩 Nm
        W_axle = car_speed / self.R_wheel * self.G_f  # 驱动电机转速 rad/s
        P_axle = T_axle * W_axle  # 驱动电机功率 w

        # 发动机
        if P_axle <= 0:
            P = 0
        P = P if P <= 85000 else 85000 # 发动机最大功率限制
        T_eng = self.find_te(P) if P >= 20093 else 0 # 发动机最小功率限制
        W_eng = self.find_we(P) if P >= 20093 else 0 # 发动机最小功率限制
        T_eng = self.Eng_t_max(W_eng) if T_eng > self.Eng_t_max(W_eng) else T_eng
        T_eng = 0 if T_eng < 0 else T_eng

        # 发电机
        W_gen, T_gen = W_eng, -T_eng
        T_gen = T_gen if T_gen > -self.Gen_t_max(W_gen) else -self.Gen_t_max(W_gen) # 发电机最大转矩限制
        T_eng = T_eng if T_gen != -self.Gen_t_max(W_gen) else -T_gen # 发动机最大转矩限制
        eff_g = self.Gen_eff(W_gen, T_gen)
        P_gen = T_gen * W_gen * eff_g # 发电机输出功率

        # 驱动电机
        W_mot = W_axle
        if T_axle > 0:# 驱动
            T_mot = T_axle if T_axle < self.Mot_t_max(W_axle) else self.Mot_t_max(W_axle)
        else: # 制动
            T_mot = T_axle if T_axle > self.Mot_t_min(W_axle) else self.Mot_t_min(W_axle)
        P_mot = W_mot * T_mot # 驱动/制动功率
        eff_m = self.Mot_eff(W_mot, T_mot)# 驱动电机效率
        if eff_m < 0.80:
            eff_m = 0.85
        P_mot = P_mot / eff_m if T_axle > 0 else P_mot * eff_m # 驱动电机功率

        # 电池
        P_batt = P_gen + P_mot # 电池功率
        if P_batt>0: # 电池放电
            r = self.R_dis(soc)
        else:   # 电池充电
            r = self.R_chg(soc)
        V_batt = self.V_oc(soc) # 电池电压
        e_batt = 1 if P_batt>0 else 0.98 # 电池效率
        if V_batt**2 - 4*r*P_batt < 0: # 超出电池最大电流
            P_gen_reg = P_gen + P_batt - V_batt**2/(4*r) # 启动发电机
            P_eng = P_gen_reg / eff_g # 发电机输出功率
            W_eng = W_eng if W_eng != 0  else 1500*self.RPM_2_rads
            T_eng = P_eng / W_eng
            W_gen = W_eng
            T_gen = T_eng
            
        # 能量消耗计算
        eff_e = self.Eng_eff(W_eng, T_eng)# 发动机效率
        P_eng = T_eng * W_eng # 发动机输出功率
        P_eng_for_fuel = T_eng * W_eng / eff_e
        m_fuel = P_eng_for_fuel / 42500000 # 发动机燃油消耗 g
        v_fuel = m_fuel / 0.72 # 发动机燃油消耗 L
        price_fuel = v_fuel*8.64 # 油价
        price_elec = P_batt / 0.8 / 1000 / 3600 * 0.97 # 电价 功率——效率——kWh——电价
        
        RMB_cost = (price_fuel + price_elec) # 钱
        
        # 电量更新
        if V_batt**2 - 4*r*P_batt + 1e-10 >= 0: # 电流
            I_batt = e_batt * ( V_batt - np.sqrt(V_batt**2 - 4*r*P_batt+1e-10))/(2*r)
        else:
            I_batt = e_batt * V_batt /(2*r)
        soc_ = - I_batt / (self.Q_batt) + soc # SOC
        soc = soc_
        if soc > 1:
            soc = 1.0
        
        out = {}
        out['P'] = P                                        # 发动机控制输入 w
                
        out['P_axle'] = P_axle                        # 车轮驱动功率 w
        out['T_axle'] = T_axle * self.G_f         # 车轮驱动转矩 Nm
        out['W_axle'] = W_axle / self.G_f      # 车轮驱动转速 rad/s
        
        out['P_mot'] = P_mot                       # 驱动电机功率 w
        out['T_mot'] = T_mot                       # 驱动电机转矩 Nm
        out['W_mot'] = W_mot                    # 驱动电机转速 rad/s
        out['eff_m'] = eff_m                         # 驱动电机效率 η
        
        out['P_eng'] = P_eng                        # 发动机功率 w
        out['T_eng'] = T_eng                        # 发动机转矩 Nm
        out['W_eng'] = W_eng                     # 发动机转速 rad/s
        out['eff_e'] = eff_e                            # 发动机效率 η
        
        out['P_gen'] = P_gen                        # 发电机功率 w
        out['T_gen'] = T_gen                        # 发电机转矩 Nm
        out['W_gen'] = W_gen                     # 发电机转速 rad/s 
        out['eff_g'] = eff_g                           # 发电机效率 η
       
        out['P_batt'] = P_batt                       # 电池功率 w
        
        out['price_fuel'] = price_fuel            # 油钱
        out['price_elec'] = price_elec           # 电费
        
        return out, RMB_cost, soc
        
#SHEV = series_HEV()    
#out,cost, soc = SHEV.run(2, 0.2, 5000, 0.5)
