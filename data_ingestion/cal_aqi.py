import math
import numpy as np
import pandas as pd

# ref: https://www.airnow.gov/sites/default/files/2020-05/aqi-technical-assistance-document-sept2018.pdf
class AQICalculator:
    """
    空气质量指数(AQI)计算器类
    
    根据EPA标准计算AQI值，支持多种污染物：
    - PM2.5_24h (细颗粒物24小时平均)
    - PM10_24h (可吸入颗粒物24小时平均)
    - O3_8h (臭氧8小时平均)
    - O3_1h (臭氧1小时平均) 
    - CO_8h (一氧化碳8小时平均)
    - SO2_1h (二氧化硫1小时平均)
    - SO2_24h (二氧化硫24小时平均)
    - NO2_1h (二氧化氮1小时平均)
    """
    
    # 所有污染物断点表
    BREAKPOINTS = {
        "PM2.5_24h": [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ],
        "PM10_24h": [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500)
        ],
        "O3_8h": [
            (0.000, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300)
            # 注意：8小时臭氧不用于计算高于300的AQI值
        ],
        "O3_1h": [
            # 注意：1小时臭氧仅用于计算AQI值≥101
            (0.125, 0.164, 101, 150),
            (0.165, 0.204, 151, 200),
            (0.205, 0.404, 201, 300),
            (0.405, 0.504, 301, 400),
            (0.505, 0.604, 401, 500)
        ],
        "CO_8h": [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500)
        ],
        "SO2_1h": [
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            # 注意：1小时SO2不用于计算高于200的AQI值
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500)
        ],
        "SO2_24h": [
            # 注意：24小时SO2仅用于计算AQI值≥201
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500)
        ],
        "NO2_1h": [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500)
        ]
    }

    # 截断规则
    TRUNCATE_RULES = {
        "PM2.5_24h": lambda x: round(x, 1),  # 截断到1位小数
        "PM10_24h": lambda x: int(x),        # 截断到整数
        "O3_8h": lambda x: round(x, 3),      # 截断到3位小数
        "O3_1h": lambda x: round(x, 3),      # 截断到3位小数
        "CO_8h": lambda x: round(x, 1),      # 截断到1位小数
        "SO2_1h": lambda x: int(x),          # 截断到整数
        "SO2_24h": lambda x: int(x),         # 截断到整数
        "NO2_1h": lambda x: int(x)           # 截断到整数
    }
    
    # AQI类别
    AQI_CATEGORIES = [
        (0, 50, "优", "空气质量令人满意，基本无空气污染"),
        (51, 100, "良", "空气质量可接受，某些污染物可能对极少数异常敏感人群健康有较弱影响"),
        (101, 150, "轻度污染", "敏感人群症状有轻度加剧，健康人群出现刺激症状"),
        (151, 200, "中度污染", "进一步加剧敏感人群症状，可能对健康人群心脏、呼吸系统有影响"),
        (201, 300, "重度污染", "健康影响显著加剧，可能对全体人群产生较强烈的健康影响"),
        (301, 500, "危险", "健康警报，所有人群的健康都会受到危害")
    ]
    
    def __init__(self, custom_breakpoints=None, custom_truncate_rules=None):
        """
        初始化AQI计算器
        
        参数:
            custom_breakpoints (dict, optional): 自定义断点表
            custom_truncate_rules (dict, optional): 自定义截断规则
        """
        self.breakpoints = custom_breakpoints or self.BREAKPOINTS
        self.truncate_rules = custom_truncate_rules or self.TRUNCATE_RULES
    
    def calculate_single_aqi(self, pollutant, concentration):
        """
        计算单个污染物的AQI值
        
        参数:
            pollutant (str): 污染物名称
            concentration (float): 污染物浓度
            
        返回:
            tuple: (截断后的浓度值, AQI值, 污染物类型)
        """
        if pollutant not in self.breakpoints:
            return None, None, pollutant
            
        truncated = self.truncate_rules[pollutant](concentration)
        aqi = self._calculate_aqi(truncated, self.breakpoints[pollutant])
        return truncated, aqi, pollutant
    
    def _calculate_aqi(self, Cp, breakpoints):
        """
        根据污染物浓度和断点表计算 AQI
        
        参数:
            Cp (float): 污染物浓度
            breakpoints (list): 断点表
            
        返回:
            int: AQI值，如果无法计算则返回None
        """
        for BP_Lo, BP_Hi, I_Lo, I_Hi in breakpoints:
            if BP_Lo <= Cp <= BP_Hi:
                # 使用EPA公式计算AQI: Ip = ((IHi-ILo)/(BPHi-BPLo)) * (Cp-BPLo) + ILo
                aqi = (I_Hi - I_Lo) / (BP_Hi - BP_Lo) * (Cp - BP_Lo) + I_Lo
                return round(aqi)  # 四舍五入到最接近的整数
        return None
    
    def calculate_overall_aqi(self, concentrations):
        """
        计算所有污染物的 AQI，并返回最大值和详情
        
        参数:
            concentrations (dict): 污染物名称和浓度的字典
            
        返回:
            tuple: (综合AQI值, 主要污染物, 详细结果字典)
        """
        results = {}
        aqi_values = []
        
        # 特殊处理臭氧
        o3_8h = concentrations.get("O3_8h")
        o3_1h = concentrations.get("O3_1h")
        
        if o3_8h is not None and o3_1h is not None:
            # 如果同时有8小时和1小时臭氧值，需要比较
            truncated_8h, aqi_8h, _ = self.calculate_single_aqi("O3_8h", o3_8h)
            
            # 只有当1小时臭氧值 >= 0.125 ppm时才计算
            if o3_1h >= 0.125:
                truncated_1h, aqi_1h, _ = self.calculate_single_aqi("O3_1h", o3_1h)
                
                # 取两者中的最大值
                if aqi_8h is not None and aqi_1h is not None:
                    if aqi_1h > aqi_8h:
                        results["O3"] = (truncated_1h, aqi_1h, "O3_1h")
                        aqi_values.append((aqi_1h, "O3_1h"))
                    else:
                        results["O3"] = (truncated_8h, aqi_8h, "O3_8h")
                        aqi_values.append((aqi_8h, "O3_8h"))
                elif aqi_8h is not None:
                    results["O3"] = (truncated_8h, aqi_8h, "O3_8h")
                    aqi_values.append((aqi_8h, "O3_8h"))
                elif aqi_1h is not None:
                    results["O3"] = (truncated_1h, aqi_1h, "O3_1h")
                    aqi_values.append((aqi_1h, "O3_1h"))
            elif aqi_8h is not None:
                results["O3"] = (truncated_8h, aqi_8h, "O3_8h")
                aqi_values.append((aqi_8h, "O3_8h"))
        elif o3_8h is not None:
            truncated_8h, aqi_8h, _ = self.calculate_single_aqi("O3_8h", o3_8h)
            if aqi_8h is not None:
                results["O3"] = (truncated_8h, aqi_8h, "O3_8h")
                aqi_values.append((aqi_8h, "O3_8h"))
        elif o3_1h is not None and o3_1h >= 0.125:
            truncated_1h, aqi_1h, _ = self.calculate_single_aqi("O3_1h", o3_1h)
            if aqi_1h is not None:
                results["O3"] = (truncated_1h, aqi_1h, "O3_1h")
                aqi_values.append((aqi_1h, "O3_1h"))
        
        # 特殊处理二氧化硫
        so2_1h = concentrations.get("SO2_1h")
        so2_24h = concentrations.get("SO2_24h")
        
        if so2_1h is not None:
            truncated_1h, aqi_1h, _ = self.calculate_single_aqi("SO2_1h", so2_1h)
            
            # 如果1小时SO2值 >= 305 ppb，需要检查24小时值
            if so2_1h >= 305 and so2_24h is not None:
                truncated_24h, aqi_24h, _ = self.calculate_single_aqi("SO2_24h", so2_24h)
                
                # 如果24小时值可以计算AQI，则取两者中的最大值
                if aqi_24h is not None and aqi_1h is not None:
                    if aqi_24h > aqi_1h:
                        results["SO2"] = (truncated_24h, aqi_24h, "SO2_24h")
                        aqi_values.append((aqi_24h, "SO2_24h"))
                    else:
                        results["SO2"] = (truncated_1h, aqi_1h, "SO2_1h")
                        aqi_values.append((aqi_1h, "SO2_1h"))
                elif aqi_1h is not None:
                    results["SO2"] = (truncated_1h, aqi_1h, "SO2_1h")
                    aqi_values.append((aqi_1h, "SO2_1h"))
            elif aqi_1h is not None:
                # 如果1小时值 < 305 ppb或没有24小时值，直接使用1小时值
                results["SO2"] = (truncated_1h, aqi_1h, "SO2_1h")
                aqi_values.append((aqi_1h, "SO2_1h"))
        
        # 处理其他污染物
        for pollutant, value in concentrations.items():
            if pollutant in ["O3_8h", "O3_1h", "SO2_1h", "SO2_24h"]:
                continue  # 已经特殊处理过
                
            if pollutant in self.breakpoints and value is not None:
                truncated, aqi, _ = self.calculate_single_aqi(pollutant, value)
                if aqi is not None:
                    results[pollutant] = (truncated, aqi, pollutant)
                    aqi_values.append((aqi, pollutant))
        
        if not aqi_values:
            return None, None, {}
            
        # 找出AQI最大的污染物
        max_aqi, main_pollutant = max(aqi_values, key=lambda x: x[0])
        
        return max_aqi, main_pollutant, results
    
    def get_aqi_category(self, aqi):
        """
        根据AQI值返回空气质量类别
        
        参数:
            aqi (int): AQI值
            
        返回:
            tuple: (类别名称, 类别描述)
        """
        if aqi is None:
            return "未知", "无法计算AQI值"
            
        for low, high, category, description in self.AQI_CATEGORIES:
            if low <= aqi <= high:
                return category, description
                
        return "超标", "AQI值超出正常范围"
    
    def calculate_nowcast_pm25(self, hourly_data, hours=12):
        """
        计算PM2.5的NowCast值
        
        参数:
            hourly_data (list): 最近12小时的PM2.5浓度数据，最新的在最后
            hours (int): 使用的小时数，默认为12
            
        返回:
            float: NowCast值
        """
        if len(hourly_data) < hours:
            return None  # 数据不足
            
        # 确保使用最近的hours个小时数据
        recent_data = hourly_data[-hours:]
        
        # 检查最近3小时是否至少有2个有效值
        recent_3h = recent_data[-3:]
        valid_recent = sum(1 for x in recent_3h if x is not None)
        if valid_recent < 2:
            return None  # 最近3小时数据不足
        
        # 移除None值
        valid_data = [x for x in recent_data if x is not None]
        if not valid_data:
            return None
        
        # 计算最小值和最大值
        min_val = min(valid_data)
        max_val = max(valid_data)
        
        # 计算变化率
        rate_of_change = (max_val - min_val) / max_val if max_val > 0 else 0
        
        # 计算权重因子
        weight_factor = 1 - rate_of_change
        weight_factor = max(0.5, weight_factor)  # 权重因子不低于0.5
        
        # 计算加权平均值
        total_weight = 0
        weighted_sum = 0
        
        for i, val in enumerate(recent_data):
            if val is not None:
                hour_weight = weight_factor ** i
                weighted_sum += val * hour_weight
                total_weight += hour_weight
        
        if total_weight > 0:
            nowcast = weighted_sum / total_weight
            return nowcast
        else:
            return None
    
    def calculate_nowcast_pm10(self, hourly_data):
        """
        计算PM10的NowCast值，与PM2.5使用相同的方法
        """
        return self.calculate_nowcast_pm25(hourly_data)
    
    def calculate_nowcast_ozone(self, hourly_data_1h, days=14):
        """
        计算臭氧的NowCast值
        
        参数:
            hourly_data_1h (list): 最近两周的1小时臭氧浓度数据
            days (int): 使用的天数，默认为14
            
        返回:
            float: 臭氧NowCast值
        """
        # 臭氧NowCast方法较复杂，需要参考EPA的R代码实现
        # https://github.com/USEPA/O3-NowCast/tree/master
        # 此处提供一个简化版本
        if len(hourly_data_1h) < 24:
            return None  # 数据不足
            
        # 使用最近24小时的数据
        recent_data = hourly_data_1h[-24:]
        
        # 计算滑动8小时平均值
        eight_hour_avgs = []
        for i in range(17):  # 24-8+1
            window = recent_data[i:i+8]
            if None not in window:
                eight_hour_avgs.append(sum(window) / 8)
        
        if not eight_hour_avgs:
            return None
            
        # 返回最大8小时平均值
        return max(eight_hour_avgs)


# 示例使用
if __name__ == "__main__":
    # 创建AQI计算器实例
    calculator = AQICalculator()
    
    # 示例：传入多个污染物的原始浓度值
    input_data = {
        "PM2.5_24h": 35.9,
        "PM10_24h": 155,
        "O3_8h": 0.078,
        "O3_1h": 0.162,  # 高于0.125，会参与计算
        "CO_8h": 8.4,
        "SO2_1h": 140,
        "NO2_1h": 56
    }
    
    overall_aqi, main_pollutant, detail = calculator.calculate_overall_aqi(input_data)
    
    # 输出
    print(f"综合 AQI：{overall_aqi}")
    category, description = calculator.get_aqi_category(overall_aqi)
    print(f"空气质量类别：{category} - {description}")
    print(f"主要污染物：{main_pollutant}")
    
    for pollutant, (trunc, aqi, poll_type) in detail.items():
        print(f"{pollutant}: 截断浓度 = {trunc}, AQI = {aqi}, 类型 = {poll_type}")
    
    # 测试NowCast计算
    pm25_hourly = [10.5, 12.8, 14.2, 15.6, 17.0, 18.5, 19.2, 18.7, 17.9, 16.5, 15.2, 14.8]
    nowcast_pm25 = calculator.calculate_nowcast_pm25(pm25_hourly)
    print(f"\nPM2.5 NowCast值: {nowcast_pm25:.1f}")
