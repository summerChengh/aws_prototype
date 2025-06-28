#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import argparse
import sys
import gzip
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, date
from utils import load_csv_file, find_data_files, preprocess_datetime_column, sort_by_datetime, load_and_merge_data_files, perform_time_resampling
# 导入OpenAQProcessor类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.openaq_feats import OpenAQProcessor


def resample_time_series(df, time_column='datetime', time_freq='1H', agg_method='mean'):
    """
    按指定时间频率重采样时间序列数据
    
    参数:
    df (pandas.DataFrame): 输入数据
    time_column (str): 时间列名
    time_freq (str): 时间频率，如'1H'表示1小时，'1D'表示1天
    agg_method (str or dict): 聚合方法，可以是字符串或字典
    
    返回:
    pandas.DataFrame: 重采样后的数据
    """
    if df is None or df.empty:
        return df
    
    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        df[time_column] = pd.to_datetime(df[time_column])
    
    # 设置时间索引
    df_indexed = df.set_index(time_column)
    
    # 按时间频率分组并聚合
    if isinstance(agg_method, str):
        # 如果是简单的字符串聚合函数
        resampled_df = df_indexed.groupby(pd.Grouper(freq=time_freq)).agg(agg_method)
    else:
        # 如果是字典形式的聚合函数
        resampled_df = df_indexed.groupby(pd.Grouper(freq=time_freq)).agg(agg_method)
    
    # 重置索引
    resampled_df = resampled_df.reset_index()
    
    return resampled_df

def download_data_from_s3(location_id, data_dir, start_date, end_date):
    """
    从AWS S3下载指定日期范围内的数据文件
    只下载指定日期范围内的文件，而不是整个月份的数据
    
    参数:
    location_id (str): 位置ID
    data_dir (str): 本地数据目录
    start_date (str): 起始日期，格式为YYYYMMDD
    end_date (str): 结束日期，格式为YYYYMMDD
    
    返回:
    bool: 是否成功下载数据
    """
    if not start_date or not end_date:
        print("错误: 下载数据需要指定起始日期和结束日期")
        return False
    
    try:
        # 解析日期范围
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        # 生成日期范围内的所有日期
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt)
            current_dt += timedelta(days=1)
        
        # 按年月组织日期
        dates_by_year_month = {}
        for dt in date_list:
            year = dt.year
            month = dt.month
            day = dt.day
            
            if (year, month) not in dates_by_year_month:
                dates_by_year_month[(year, month)] = []
            
            dates_by_year_month[(year, month)].append(day)
        
        download_success = False
        downloaded_files = []
        
        # 对每个年月组合执行下载
        for (year, month), days in sorted(dates_by_year_month.items()):
            year_str = str(year)
            month_str = f"{month:02d}"
            
            # 确保本地目录存在
            local_dir = os.path.join(data_dir, f"locationid={location_id}/year={year_str}/month={month_str}")
            os.makedirs(local_dir, exist_ok=True)
            
            # 对每一天分别下载
            for day in sorted(days):
                day_str = f"{day:02d}"
                file_name = f"location-{location_id}-{year_str}{month_str}{day_str}.csv.gz"
                
                # 检查本地文件是否已存在
                local_file = os.path.join(local_dir, file_name)
                if os.path.exists(local_file):
                    print(f"文件已存在，跳过下载: {file_name}")
                    downloaded_files.append(local_file)
                    download_success = True
                    continue
                
                # 构建S3路径和本地路径
                s3_file = f"s3://openaq-data-archive/records/csv.gz/locationid={location_id}/year={year_str}/month={month_str}/{file_name}"
                # 检查S3文件是否存在
                check_cmd = [
                    "aws", "s3", "ls",
                    "--no-sign-request",
                    s3_file
                ]
                check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                if check_result.returncode != 0 or not check_result.stdout.strip():
                    print(f"S3文件不存在，跳过: {s3_file}")
                    continue
                print(f"正在从S3下载文件: {s3_file} -> {local_file}")
                
                # 执行AWS CLI命令下载单个文件
                cmd = [
                    "aws", "s3", "cp",
                    "--no-sign-request",
                    s3_file,
                    local_file
                ]
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"下载完成: {file_name}")
                    downloaded_files.append(local_file)
                    download_success = True
                except subprocess.CalledProcessError as e:
                    print(f"下载失败: {file_name}")
                    print(f"错误输出: {e.stderr}")
        
        print(f"总共下载了 {len(downloaded_files)} 个文件")
        return download_success
    
    except Exception as e:
        print(f"下载数据时出错: {e}")
        return False


def fetch_data_from_s3_by_year(location_id, data_dir, year, check_existing=True):
    """
    从AWS S3按年下载数据文件
    
    参数:
    location_id (str): 位置ID
    data_dir (str): 本地数据目录
    year (int): 年份，如2024
    check_existing (bool): 是否检查已存在的文件，默认True
    
    返回:
    tuple: (是否成功下载数据, 下载的文件列表)
    """
    try:
        # 格式化年份
        year_str = str(year)
        
        # 确保本地目录存在
        location_dir = os.path.join(data_dir, f"locationid={location_id}/year={year_str}")
        os.makedirs(location_dir, exist_ok=True)
        
        # 构建S3路径
        s3_base_path = f"s3://openaq-data-archive/records/csv.gz/locationid={location_id}/year={year_str}/"
        
        print(f"正在从S3获取 {year_str} 年的月份列表...")
        
        # 获取所有月份目录
        list_cmd = [
            "aws", "s3", "ls",
            "--no-sign-request",
            s3_base_path
        ]
        
        try:
            result = subprocess.run(list_cmd, check=True, capture_output=True, text=True)
            month_dirs = []
            
            # 解析输出获取月份目录
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1].startswith("month="):
                    month_dir = parts[-1].strip('/')
                    month = month_dir.split('=')[1]
                    month_dirs.append(month)
            
            if not month_dirs:
                print(f"未找到 {year_str} 年的月份目录")
                return False, []
            
            print(f"找到 {len(month_dirs)} 个月份目录: {', '.join(month_dirs)}")
            
            download_success = False
            downloaded_files = []
            
            # 对每个月份执行下载
            for month in sorted(month_dirs):
                month_path = f"{s3_base_path}month={month}/"
                local_month_dir = os.path.join(location_dir, f"month={month}")
                os.makedirs(local_month_dir, exist_ok=True)
                
                print(f"正在下载 {year_str} 年 {month} 月的数据...")
                
                # 列出该月的所有文件
                list_files_cmd = [
                    "aws", "s3", "ls",
                    "--no-sign-request",
                    month_path
                ]
                
                files_result = subprocess.run(list_files_cmd, check=True, capture_output=True, text=True)
                files_to_download = []
                
                # 解析输出获取文件列表
                for line in files_result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 4:
                        file_name = parts[-1]
                        if file_name.endswith(".csv.gz"):
                            files_to_download.append(file_name)
                
                if not files_to_download:
                    print(f"月份 {month} 没有找到文件")
                    continue
                
                print(f"找到 {len(files_to_download)} 个文件需要下载")
                
                # 下载每个文件
                for file_name in files_to_download:
                    local_file = os.path.join(local_month_dir, file_name)
                    
                    # 检查本地文件是否已存在
                    if check_existing and os.path.exists(local_file):
                        print(f"文件已存在，跳过下载: {file_name}")
                        downloaded_files.append(local_file)
                        download_success = True
                        continue
                    
                    s3_file = f"{month_path}{file_name}"
                    print(f"正在从S3下载文件: {s3_file} -> {local_file}")
                    
                    # 执行AWS CLI命令下载单个文件
                    cmd = [
                        "aws", "s3", "cp",
                        "--no-sign-request",
                        s3_file,
                        local_file
                    ]
                    
                    try:
                        dl_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        print(f"下载完成: {file_name}")
                        downloaded_files.append(local_file)
                        download_success = True
                    except subprocess.CalledProcessError as e:
                        print(f"下载失败: {file_name}")
                        print(f"错误输出: {e.stderr}")
            
            print(f"总共下载了 {len(downloaded_files)} 个文件")
            return download_success, downloaded_files
            
        except subprocess.CalledProcessError as e:
            print(f"获取月份列表失败: {e.stderr}")
            return False, []
            
    except Exception as e:
        print(f"按年下载数据时出错: {e}")
        return False, []


def process_with_openaq_processor(df):
    """
    使用OpenAQProcessor处理数据
    
    参数:
    df (pandas.DataFrame): 输入数据
    
    返回:
    tuple: (处理后的数据DataFrame, 特征数据DataFrame, OpenAQProcessor实例)
    """
    if df is None or df.empty:
        print("警告: 数据为空，无法使用OpenAQProcessor处理")
        return None, None, None
    
    # 使用OpenAQProcessor处理数据
    processor = OpenAQProcessor()
    
    # 加载数据
    processor.df = df.copy()
    
    # 解析datetime列
    try:
        print("尝试解析datetime列...")
        processor.parse_datetime()
    except Exception as e:
        print(f"解析datetime列时出错: {e}")
        print("尝试修复datetime格式并重新解析...")
        
        try:
            # 手动解析datetime列
            if 'datetime' in processor.df.columns:
                # 确保datetime列是字符串类型
                processor.df['datetime'] = processor.df['datetime'].astype(str)
                
                # 标准化格式
                processor.df['datetime'] = processor.df['datetime'].str.replace(' ', 'T')
                
                # 移除时区信息
                timezone_pattern = r'([+-]\d{2}:?\d{2}|\.\d+|Z)$'
                processor.df['datetime'] = processor.df['datetime'].str.replace(timezone_pattern, '', regex=True)
                
                # 手动提取日期和小时
                processor.df['date'] = pd.to_datetime(processor.df['datetime'].str.split('T').str[0]).dt.date
                
                # 提取小时
                processor.df['hour'] = pd.to_numeric(
                    processor.df['datetime'].str.split('T').str[1].str.split(':').str[0],
                    errors='coerce'
                )
                
                print("手动解析datetime列完成")
            else:
                print("数据中没有datetime列，无法解析")
                return None, None, None
        except Exception as e2:
            print(f"手动解析datetime列也失败: {e2}")
            return None, None, None
    
    # 分析参数
    processor.analyze_parameters()
    
    # 填充缺失的小时
    try:
        processor.fill_missing_hours()
    except Exception as e:
        print(f"填充缺失小时时出错: {e}")
        print("跳过填充缺失小时步骤...")
    
    # 获取处理后的数据
    try:
        processed_df = processor.get_processed_data()
        print(f"处理后数据形状: {processed_df.shape}")
    except Exception as e:
        print(f"获取处理后的数据时出错: {e}")
        processed_df = processor.df
        print("使用原始处理数据继续...")
    
    # 计算滑动窗口特征
    features_df = None
    try:
        processor.calculate_sliding_window_features()
        
        # 获取特征数据
        features_df = processor.get_aggregated_data()
        print(f"特征数据形状: {features_df.shape if features_df is not None else 'None'}")
    except Exception as e:
        print(f"计算滑动窗口特征时出错: {e}")
    
    return processed_df, features_df, processor


def process_aq_data_and_extract_features(location_id, data_dir=None, output_dir=None, start_date=None, end_date=None, time_freq=None, agg_method='mean', download_missing=False):
    """
    处理空气质量数据并提取特征
    
    参数:
    location_id (str): 位置ID
    data_dir (str): 数据根目录
    output_dir (str, optional): 输出目录，默认为当前目录
    start_date (str, optional): 起始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    time_freq (str, optional): 时间重采样频率，如'1H'表示1小时，'1D'表示1天，默认为None不重采样
    agg_method (str or dict): 聚合方法，可以是字符串或字典，默认为'mean'
    download_missing (bool): 如果数据不存在，是否从AWS S3下载
    
    返回:
    tuple: (合并后的原始数据DataFrame, 提取特征后的DataFrame)
    """
    # 步骤1: 确保输出目录和数据目录存在
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "data/opendb-aq")
    os.makedirs(data_dir, exist_ok=True)
    
    # 构建输出文件名前缀
    date_range = f"{start_date}_to_{end_date}" if start_date and end_date else "all_data"
    freq_prefix = f"freq_{time_freq}_" if time_freq else ""
    file_prefix = f"loc_{location_id}_{date_range}_"
    
    # 步骤2: 查找并下载数据文件
    csv_files = find_data_files(data_dir, location_id, start_date, end_date)
    
    if not csv_files and download_missing and start_date and end_date:
        print(f"未找到匹配的CSV文件，尝试从AWS S3下载...")
        download_success = download_data_from_s3(location_id, data_dir, start_date, end_date)
        
        if download_success:
            csv_files = find_data_files(data_dir, location_id, start_date, end_date)
            if not csv_files:
                print("下载后仍未找到匹配的CSV文件，可能S3上没有该日期范围的数据")
                return None, None, None
        else:
            print("从AWS S3下载数据失败")
            return None, None, None
    elif not csv_files:
        print(f"未找到匹配的CSV文件")
        return None, None, None
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 步骤3: 加载并合并数据文件
    merged_df = load_and_merge_data_files(csv_files)
    if merged_df is None:
        return None, None, None
    
    # 步骤4: 预处理datetime列
    merged_df = preprocess_datetime_column(merged_df)
    
    # 步骤5: 按datetime排序
    merged_df = sort_by_datetime(merged_df)
    
    # 保存合并后的原始数据
    raw_data_file = os.path.join(output_dir, f"raw_data_{file_prefix}.csv")
    merged_df.to_csv(raw_data_file, index=False)
    print(f"合并后的原始数据已保存到: {raw_data_file}")
    
    # 步骤6: 如果指定了时间频率，进行时间重采样
    if time_freq:
        merged_df = perform_time_resampling(merged_df, time_freq, agg_method)
        
        # 保存重采样后的数据
        resampled_file = os.path.join(output_dir, f"{freq_prefix}resampled_{file_prefix}.csv")
        merged_df.to_csv(resampled_file, index=False)
        print(f"时间重采样后的数据已保存到: {resampled_file}")
    
    # 步骤7: 使用OpenAQProcessor处理数据
    processed_df, features_df, processor = process_with_openaq_processor(merged_df)
    
    # 步骤8: 保存处理后的数据
    if processed_df is not None:
        hourly_file = os.path.join(output_dir, f"{freq_prefix}hourly_data_{file_prefix}.csv")
        processed_df.to_csv(hourly_file, index=False)
        print(f"填充小时后的数据已保存到: {hourly_file}")
    
    # 步骤9: 保存特征数据
    if features_df is not None:
        features_file = os.path.join(output_dir, f"{freq_prefix}features_{file_prefix}.csv")
        features_df.to_csv(features_file, index=False)
        print(f"提取的特征数据已保存到: {features_file}")
    
    return merged_df, processed_df, features_df

def main():
    parser = argparse.ArgumentParser(description='处理空气质量数据并提取特征')
    parser.add_argument('--location_id', type=str, required=True, help='位置ID')
    parser.add_argument('--data_dir', type=str, help='数据根目录，包含locationid={location_id}子目录，默认为./data/opendb-aq')
    parser.add_argument('--output_dir', type=str, help='输出目录，默认为当前目录')
    parser.add_argument('--start_date', type=str, help='起始日期，格式为YYYYMMDD，如20240101')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD，如20240131')
    parser.add_argument('--time_freq', type=str, help='时间重采样频率，如1H表示1小时，1D表示1天，不指定则不重采样')
    parser.add_argument('--agg_method', type=str, default='mean', help='聚合方法，默认为mean，可选值：mean, median, max, min, sum')
    parser.add_argument('--download', action='store_true', help='如果数据不存在，从AWS S3下载')
    parser.add_argument('--check_incomplete', action='store_true', help='检查并下载不完整的月份数据')
    parser.add_argument('--start_year', type=int, help='检查不完整月份的起始年份')
    parser.add_argument('--end_year', type=int, help='检查不完整月份的结束年份')
    parser.add_argument('--start_month', type=int, help='检查不完整月份的起始月份')
    parser.add_argument('--end_month', type=int, help='检查不完整月份的结束月份')
    parser.add_argument('--download_year', type=int, help='下载指定年份的所有数据')
    parser.add_argument('--force_download', action='store_true', help='强制下载，不检查本地是否已存在文件')
    
    args = parser.parse_args()
    
    # 设置默认数据目录
    data_dir = args.data_dir if args.data_dir else os.path.join(os.getcwd(), "data/opendb-aq")
    
    # 按年下载数据
    if args.download_year is not None:
        print(f"正在下载位置ID {args.location_id} 的 {args.download_year} 年数据...")
        success, files = fetch_data_from_s3_by_year(
            location_id=args.location_id,
            data_dir=data_dir,
            year=args.download_year,
            check_existing=not args.force_download
        )
        
        status = "成功" if success else "失败"
        print(f"\n下载结果: {status}, 文件数: {len(files)}")
        return
    
    # 检查是否需要下载不完整月份
    if args.check_incomplete:
        print(f"检查并下载位置ID {args.location_id} 的不完整月份数据...")
        results = check_and_download_incomplete_months(
            location_id=args.location_id,
            data_dir=data_dir,
            start_year=args.start_year,
            end_year=args.end_year,
            start_month=args.start_month,
            end_month=args.end_month
        )
        
        print("\n下载结果汇总:")
        for (year, month), result in sorted(results.items()):
            status = "成功" if result['success'] else "失败"
            print(f"{year}-{month:02d}: {status}, 文件数: {result['files_count']}")
    else:
        # 处理聚合方法
        agg_method = args.agg_method
        if agg_method not in ['mean', 'median', 'max', 'min', 'sum']:
            print(f"警告: 不支持的聚合方法 {agg_method}，使用默认值 mean")
            agg_method = 'mean'
        
        process_aq_data_and_extract_features(
            args.location_id,
            data_dir,
            args.output_dir,
            args.start_date,
            args.end_date,
            args.time_freq,
            agg_method,
            args.download
        )


if __name__ == '__main__':
    main() 