#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os
import re
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Union, Any, Tuple, Set

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAQProcessor:
    """
    OpenAQ数据处理器，用于处理和分析OpenAQ空气质量数据
    """
    
    def __init__(self, fill_value=np.nan):
        """
        初始化OpenAQ处理器
        
        参数:
        fill_value: 填充缺失值使用的默认值，默认为NaN
        """
        self.fill_value = fill_value
        self.df = None
        self.processed_df = None
        self.location_col = 'location_id'
        self.aggregated_df = None
        self.parameters_dict ={24: ["pm25", "pm10", "so2"], 8: ["o3", "co"], 1: ["o3", "so2", "no2"]}
        
        # 默认的滑动窗口大小（小时）
        self.window_sizes = [3, 6, 12, 24, 48, 72]
        
        # 默认的聚合函数
        self.agg_functions = ['mean', 'min', 'max', 'std']
        
    def load_data(self, csv_file):
        """
        加载CSV数据文件
        
        参数:
        csv_file (str): CSV文件路径
        
        返回:
        self: 支持链式调用
        """
        print(f"加载数据文件: {csv_file}")
        self.df = pd.read_csv(csv_file)
        
        # 显示原始数据信息
        print(f"原始数据形状: {self.df.shape}")
        print(f"原始数据列: {self.df.columns.tolist()}")
        
        # 检查必要的列是否存在
        required_cols = ['location_id', 'datetime', 'parameter', 'value']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")

        return self
    
    def parse_datetime(self):
        """
        从datetime列解析出小时和日期信息
        
        返回:
        OpenAQProcessor: 返回self以支持链式调用
        """
        print("从datetime列解析出时间信息...")
        
        if 'datetime' not in self.df.columns:
            print("警告: 数据中没有datetime列，无法解析时间信息")
            return self
        
        # 确保datetime列是字符串类型
        self.df['datetime'] = self.df['datetime'].astype(str)
        
        # 标准化datetime格式
        # 1. 替换空格为'T'
        self.df['datetime'] = self.df['datetime'].str.replace(' ', 'T')
        
        # 2. 处理时区信息，移除时区部分以避免混合时区问题
        # 保存原始datetime列，以便后续参考
        self.df['original_datetime'] = self.df['datetime'].copy()
        
        # 查找常见的时区模式并移除
        timezone_pattern = r'([+-]\d{2}:?\d{2}|\.\d+|Z)$'
        self.df['datetime'] = self.df['datetime'].str.replace(timezone_pattern, '', regex=True)
        
        # 3. 确保格式为 YYYY-MM-DDThh:mm:ss
        # 添加缺失的秒数
        self.df.loc[self.df['datetime'].str.count(':') == 1, 'datetime'] = self.df.loc[self.df['datetime'].str.count(':') == 1, 'datetime'] + ':00'
        
        try:
            # 尝试将datetime转换为datetime类型，设置utc=True避免混合时区警告
            self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce', utc=True)
            
            # 检查无效的datetime条目
            invalid_count = self.df['datetime'].isna().sum()
            if invalid_count > 0:
                print(f"警告: 发现 {invalid_count} 条无效的datetime记录")
            
            # 提取日期和小时信息
            self.df['date'] = self.df['datetime'].dt.date
            self.df['hour'] = self.df['datetime'].dt.hour
            
            # 提取其他日期相关特征
            self.df['day'] = self.df['date'].apply(lambda x: x.day if pd.notna(x) else None)
            self.df['month'] = self.df['date'].apply(lambda x: x.month if pd.notna(x) else None)
            self.df['year'] = self.df['date'].apply(lambda x: x.year if pd.notna(x) else None)
            self.df['dayofweek'] = self.df['datetime'].dt.dayofweek
            self.df['dayofyear'] = self.df['datetime'].dt.dayofyear
            self.df['is_weekend'] = self.df['dayofweek'].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)
            
        except Exception as e:
            print(f"使用pandas转换datetime时出错: {e}")
            print("尝试使用字符串处理方法...")
            
            # 回退到字符串处理方法
            try:
                # 确保datetime列是字符串类型
                self.df['datetime'] = self.df['datetime'].astype(str)
                
                # 从字符串中提取日期部分
                date_parts = self.df['datetime'].str.split('T').str[0]
                self.df['date'] = pd.to_datetime(date_parts, errors='coerce').dt.date
                
                # 从字符串中提取时间部分
                time_parts = self.df['datetime'].str.split('T').str[1]
                if time_parts.isna().any():
                    # 尝试其他分隔符
                    time_parts = self.df['datetime'].str.split(' ').str[1]
                
                # 提取小时
                self.df['hour'] = pd.to_numeric(time_parts.str.split(':').str[0], errors='coerce')
                
                # 提取其他日期相关特征
                self.df['day'] = self.df['date'].apply(lambda x: x.day if pd.notna(x) else None)
                self.df['month'] = self.df['date'].apply(lambda x: x.month if pd.notna(x) else None)
                self.df['year'] = self.df['date'].apply(lambda x: x.year if pd.notna(x) else None)
                
                # 计算dayofweek和dayofyear
                temp_dt = pd.to_datetime(self.df['date'], errors='coerce')
                self.df['dayofweek'] = temp_dt.dt.dayofweek
                self.df['dayofyear'] = temp_dt.dt.dayofyear
                self.df['is_weekend'] = self.df['dayofweek'].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)
                
            except Exception as e2:
                print(f"使用字符串处理方法也失败: {e2}")
                print("无法解析datetime列，时间特征提取失败")
                return self
        
        print("成功提取时间特征")
        return self
    
    def analyze_parameters(self):
        """
        分析数据中的参数
        
        返回:
        self: 支持链式调用
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        # 检查数据中的唯一参数
        unique_params = self.df['parameter'].unique()
        print(f"发现 {len(unique_params)} 个唯一的参数: {unique_params}")
        
        return self
    
    def fill_missing_hours(self):
        """
        对每个唯一的location、parameter和日期组合，填充缺失的小时
        
        返回:
        self: 支持链式调用
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if 'hour' not in self.df.columns or 'date' not in self.df.columns:
            raise ValueError("请先解析datetime")
            
        print("填充缺失的小时数据...")
        
        # 创建完整的结果DataFrame
        result_dfs = []
        
        for loc in self.df[self.location_col].unique():
            for param in self.df['parameter'].unique():
                for date in self.df['date'].unique():
                    # 筛选特定location、parameter和日期的数据
                    subset = self.df[(self.df[self.location_col] == loc) & 
                                (self.df['parameter'] == param) & 
                                (self.df['date'] == date)]
                    
                    if len(subset) <= 1:
                        continue
                    
                    # 获取当前子集的所有列
                    columns = subset.columns.tolist()
                    
                    # 创建0-23小时的完整范围
                    hours_range = pd.DataFrame({'hour': range(24)})
                    
                    # 合并现有数据与完整小时范围
                    merged = pd.merge(hours_range, subset, on='hour', how='left')
                    
                    # 填充缺失的非小时列
                    for col in columns:
                        if col not in ['hour', 'value', 'datetime']:
                            # 对于非值列，使用第一个非空值填充
                            first_valid = subset[col].iloc[0] if not subset.empty else None
                            merged[col] = merged[col].fillna(first_valid)
                    
                    # 对于值列，使用指定的默认值填充
                    merged['value'] = merged['value'].fillna(self.fill_value)
                    
                    # 重建datetime列
                    base_date = date
                    merged['datetime'] = merged.apply(
                        lambda row: datetime.combine(base_date, datetime.min.time()) + timedelta(hours=int(row['hour'])), 
                        axis=1
                    )
                           # 添加到结果列表
                    result_dfs.append(merged)
        
        # 合并所有结果
        if result_dfs:
            self.processed_df = pd.concat(result_dfs, ignore_index=True)
            
            # 按location、parameter、datetime排序
            sort_cols = [self.location_col, 'parameter', 'datetime']
            self.processed_df = self.processed_df.sort_values(sort_cols)
            
            # 重建带时区的datetime字符串表示
            if 'timezone' in self.processed_df.columns:
                self.processed_df['datetime_str'] = self.processed_df.apply(
                    lambda row: row['datetime'].strftime('%Y-%m-%dT%H:%M:%S') + row['timezone'], 
                    axis=1
                )
            
            print(f"处理后数据形状: {self.processed_df.shape}")
        else:
            print("警告: 处理后没有数据")
            self.processed_df = pd.DataFrame()
            
        return self
    
    def get_processed_data(self):
        """
        获取处理后的数据
        
        返回:
        pandas.DataFrame: 处理后的数据
        """
        if self.processed_df is None:
            raise ValueError("请先处理数据")
            
        return self.processed_df
    
    def save_data(self, output_file):
        """
        保存处理后的数据
        
        参数:
        output_file (str): 输出CSV文件路径
        
        返回:
        self: 支持链式调用
        """
        if self.processed_df is None:
            raise ValueError("请先处理数据")
            
        self.processed_df.to_csv(output_file, index=False)
        print(f"处理后的数据已保存到: {output_file}")
        
        # 显示填充统计信息
        original_count = len(self.df)
        processed_count = len(self.processed_df)
        filled_count = processed_count - original_count
        print(f"原始数据记录数: {original_count}")
        print(f"处理后数据记录数: {processed_count}")
        print(f"填充的缺失小时记录数: {filled_count}")
        
        return self
    
    def process_file(self, input_file, output_file=None):
        """
        处理单个文件的便捷方法
        
        参数:
        input_file (str): 输入CSV文件路径
        output_file (str, optional): 输出CSV文件路径，如果未提供则生成默认名称
        
        返回:
        pandas.DataFrame: 处理后的数据
        """
        # 生成默认输出文件名
        if output_file is None:
            output_file = f"processed_{os.path.basename(input_file)}"
        
        # 处理流程
        self.load_data(input_file)
        self.parse_datetime()
        self.analyze_parameters()
        self.fill_missing_hours()
        self.save_data(output_file)
        
        return self.processed_df
    
    def calculate_sliding_window_features(self, value_column='value', min_periods=None):
        """
        根据parameters_dict中定义的窗口大小计算滑动窗口平均值，
        并提取每天每个参数的最大滑动窗口平均值作为特征。
        如果某个参数在parameters_dict中不存在，则对应特征列填充NaN。
        保留location, latitude和longitude信息到特征表中。
        
        参数:
        value_column (str): 值列的名称，默认为'value'
        min_periods (int, optional): 窗口中所需的最小非NaN观测值数量，默认为窗口大小的一半
        
        返回:
        self: 支持链式调用
        """
        if self.processed_df is None:
            raise ValueError("请先处理数据并填充缺失的小时")
            
        print("根据预定义窗口大小计算滑动窗口特征...")
        
        # 获取所有唯一的location和date组合
        location_ids = self.processed_df[self.location_col].unique()
        dates = self.processed_df['date'].unique()
        
        # 创建一个空的DataFrame来存储结果
        result_df = pd.DataFrame()
        
        # 遍历每个location和date
        for loc in location_ids:
            loc_data = []
            
            # 获取该location的经纬度和位置名称
            loc_info = self.processed_df[self.processed_df[self.location_col] == loc].iloc[0:1]
            location_name = None
            latitude = None
            longitude = None
            
            # 尝试获取location名称
            if 'location' in loc_info.columns:
                location_name = loc_info['location'].iloc[0] if not loc_info.empty else None
            
            # 尝试获取经纬度
            if 'lat' in loc_info.columns:
                latitude = loc_info['lat'].iloc[0] if not loc_info.empty else None
            if 'lon' in loc_info.columns:
                longitude = loc_info['lon'].iloc[0] if not loc_info.empty else None
            
            for date in dates:
                # 创建基础行，包含location_id, location, date, latitude, longitude
                row = {
                    self.location_col: loc,
                    'date': date
                }
                
                # 添加location名称和经纬度（如果可用）
                if location_name is not None:
                    row['location'] = location_name
                if latitude is not None:
                    row['lat'] = latitude
                if longitude is not None:
                    row['lon'] = longitude
                
                # 遍历parameters_dict中定义的所有窗口大小和参数
                for window_size, parameters in self.parameters_dict.items():
                    for param in parameters:
                        param_lower = param.lower()
                        # 筛选特定location、parameter和date的数据
                        subset = self.processed_df[
                            (self.processed_df[self.location_col] == loc) & 
                            (self.processed_df['parameter'].str.lower() == param_lower) & 
                            (self.processed_df['date'] == date)
                        ].copy()
                        
                        # 特征名称
                        feature_name = f'{param_lower}_{window_size}h'
                        
                        # 如果数据不足，填充NaN
                        if len(subset) <= 1:
                            row[feature_name] = np.nan
                            continue
                        
                        # 确保数据按datetime排序
                        subset = subset.sort_values('datetime')
                        
                        # 确定min_periods
                        curr_min_periods = min_periods
                        if curr_min_periods is None:
                            curr_min_periods = max(1, window_size // 2)  # 默认为窗口大小的一半（至少为1）

                        # 将小于0的值替换为np.nan
                        subset.loc[subset[value_column] < 0, value_column] = np.nan

                        # 计算滑动窗口平均值
                        subset['sliding_avg'] = subset[value_column].rolling(
                            window=window_size, 
                            min_periods=curr_min_periods,
                            center=False  # 使用过去的值
                        ).mean()
                        
                        # 计算当天的最大滑动窗口平均值
                        max_avg = subset['sliding_avg'].max()
                        row[feature_name] = max_avg if not pd.isna(max_avg) else np.nan
                
                # 将行添加到location数据中
                loc_data.append(row)
            
            # 将location数据添加到结果DataFrame
            loc_df = pd.DataFrame(loc_data)
            result_df = pd.concat([result_df, loc_df], ignore_index=True)
        
        # 保存结果
        self.aggregated_df = result_df
        
        if not result_df.empty:
            # 按location和date排序
            sort_cols = [self.location_col, 'date']
            self.aggregated_df = self.aggregated_df.sort_values(sort_cols)
            
            # 显示结果信息
            print(f"聚合后数据形状: {self.aggregated_df.shape}")
            print(f"特征列: {[col for col in self.aggregated_df.columns if col not in [self.location_col, 'location', 'date', 'latitude', 'longitude']]}")
            
            # 显示缺失值统计
            missing_counts = self.aggregated_df.isna().sum()
            print("\n缺失值统计:")
            for col, count in missing_counts.items():
                if count > 0:
                    pct = count / len(self.aggregated_df) * 100
                    print(f"{col}: {count} ({pct:.1f}%)")
        else:
            print("警告: 聚合后没有数据")
        
        return self
    
    def get_aggregated_data(self):
        """
        获取聚合后的数据
        
        返回:
        pandas.DataFrame: 聚合后的数据
        """
        if self.aggregated_df is None:
            raise ValueError("请先计算滑动窗口特征")
            
        return self.aggregated_df
    
    def save_aggregated_data(self, output_file):
        """
        保存聚合后的数据
        
        参数:
        output_file (str): 输出CSV文件路径
        
        返回:
        self: 支持链式调用
        """
        if self.aggregated_df is None:
            raise ValueError("请先计算滑动窗口特征")
            
        self.aggregated_df.to_csv(output_file, index=False)
        print(f"聚合后的数据已保存到: {output_file}")
        
        return self


def load_and_process_openaq_data(csv_file, fill_value=np.nan):
    """
    加载OpenAQ CSV数据，从datetime列解析出hour列，
    并对每个唯一的parameter缺失的hour填充默认值
    (兼容性函数，使用OpenAQProcessor类实现)
    
    参数:
    csv_file (str): CSV文件路径
    fill_value: 填充缺失值使用的默认值，默认为NaN
    
    返回:
    pandas.DataFrame: 处理后的数据
    """
    processor = OpenAQProcessor(fill_value)
    return processor.process_file(csv_file)


def main():
    parser = argparse.ArgumentParser(description='处理OpenAQ CSV数据，填充缺失的小时数据')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, help='输出CSV文件路径')
    parser.add_argument('--fill_value', type=float, default=np.nan, help='填充缺失值使用的默认值，默认为NaN')
    parser.add_argument('--fill_zero', action='store_true', help='使用0填充缺失值')
    parser.add_argument('--calculate_window', action='store_true', help='计算滑动窗口特征')
    parser.add_argument('--agg_output', type=str, help='聚合数据输出CSV文件路径')
    
    args = parser.parse_args()
    
    # 确定填充值
    fill_value = 0.0 if args.fill_zero else args.fill_value
    
    # 创建处理器并处理数据
    processor = OpenAQProcessor(fill_value)
    processor.process_file(args.input, args.output)
    
    # 如果需要，计算滑动窗口特征
    if args.calculate_window:
        processor.calculate_sliding_window_features()
        
        # 保存聚合数据
        agg_output = args.agg_output if args.agg_output else f"aggregated_{os.path.basename(args.input)}"
        processor.save_aggregated_data(agg_output)


if __name__ == '__main__':
    main()
