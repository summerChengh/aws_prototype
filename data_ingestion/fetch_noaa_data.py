#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
import requests
from datetime import datetime, timedelta
import logging
import boto3
from botocore.client import Config
from botocore import UNSIGNED
from io import StringIO
import subprocess
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from openaq_utils import fetch_openaq_location_ids, get_location_by_id
from utils import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 数据集URL: https://registry.opendata.aws/noaa-gsod/
# 通过Amazon S3获取NOOA数据

def getNoaaDataFrame(stations, start_date, end_date):
    """
    获取多个站点在指定日期范围内的NOAA GSOD数据
    
    参数:
    stations (list): NOAA站点ID列表
    start_date (str): 开始日期，格式为 'YYYY-MM-DD'
    end_date (str): 结束日期，格式为 'YYYY-MM-DD'
    
    返回:
    pandas.DataFrame: 包含所有站点数据的合并DataFrame
    """
    logger.info(f"获取{len(stations)}个站点从{start_date}到{end_date}的数据")
    
    # 获取日期范围内的所有年份
    years = get_years_from_time_range(start_date, end_date)
    logger.info(f"需要获取的年份: {years}")
    
    # 初始化结果DataFrame
    all_data = pd.DataFrame()
    
    # 处理每个站点
    for station_idx, station_id in enumerate(stations):
        logger.info(f"处理站点 {station_idx+1}/{len(stations)}: {station_id}")
        
        # 获取每个年份的数据
        station_data = pd.DataFrame()
        for year in years:
            year_str = str(year)
            logger.info(f"获取站点 {station_id} 在 {year_str} 年的数据")
            
            # 使用已有的函数获取该站点该年份的数据
            df = getNoaaDataFrameById(station_id, year_str)
            
            if df.empty:
                logger.warning(f"站点 {station_id} 在 {year_str} 年没有数据")
                continue
                
            # 确保DATE列存在
            if 'DATE' not in df.columns:
                logger.warning(f"站点 {station_id} 的数据缺少DATE列")
                continue
                
            # 过滤日期范围
            df['DATE'] = pd.to_datetime(df['DATE'])
            mask = (df['DATE'] >= start_date) & (df['DATE'] <= end_date)
            filtered_df = df.loc[mask]
            
            if filtered_df.empty:
                logger.info(f"站点 {station_id} 在 {year_str} 年的数据在日期范围内为空")
                continue
                
            # 添加站点ID列（如果不存在）
            if 'STATION' not in filtered_df.columns:
                filtered_df['STATION'] = station_id
                
            # 合并到站点数据中
            station_data = pd.concat([station_data, filtered_df], ignore_index=True)
        
        if not station_data.empty:
            # 合并到总数据中
            all_data = pd.concat([all_data, station_data], ignore_index=True)
            logger.info(f"已添加站点 {station_id} 的 {len(station_data)} 条记录")
        else:
            logger.warning(f"站点 {station_id} 在指定日期范围内没有数据")
    
    logger.info(f"总共获取了 {len(all_data)} 条记录")
    return all_data


def getNoaaDataFrameById(stationid, year):
    """
    根据指定的站点ID和年份获取NOAA GSOD数据
    ref: https://github.com/aws-samples/aws-smsl-predict-airquality-via-weather/
    
    参数:
    stationid (str): NOAA站点ID
    year (str): 年
    
    返回:
    pandas.DataFrame: 包含NOAA数据的DataFrame
    """
    # ASDI Dataset Name: NOAA GSOD
    # ASDI Dataset URL : https://registry.opendata.aws/noaa-gsod/
    # NOAA GSOD README : https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt
    # NOAA GSOD data in S3 is organized by year and Station ID values, so this is straight-forward
    # Example S3 path format => s3://noaa-gsod-pds/{yyyy}/{stationid}.csv
    noaagsod_df = pd.DataFrame()
        
    # 构造本地文件路径
    base_dir = "./data/noaa-gsod-pds"
    file_dir = f"{base_dir}/{year}"
    file_path = f"{file_dir}/{stationid}.csv"
    
    # 确保目录存在
    os.makedirs(file_dir, exist_ok=True)
    
    # 检查是否已经存在日期特定的数据文件
    if os.path.exists(file_path):
        logger.info(f"从本地文件加载数据: {file_path}")
        return pd.read_csv(file_path)
    else:       
        # 如果本地没有数据，从S3下载
        noaagsod_bucket = 'noaa-gsod-pds'
        logger.info(f"从S3下载数据 (bucket: {noaagsod_bucket})...")
        
        try:
            s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            key = f'{year}/{stationid}.csv'
            
            # 获取S3对象
            csv_obj = s3.get_object(Bucket=noaagsod_bucket, Key=key)
            csv_string = csv_obj['Body'].read().decode('utf-8')
            
            df = pd.read_csv(StringIO(csv_string))
            df.to_csv(file_path, index=False)
            logger.info(f"已保存数据到: {file_path}")
            
            return df
                        
        except Exception as e:
            logger.error(f"下载或处理数据时出错: {str(e)}")
            return pd.DataFrame()


def download_nooa_from_s3(year):
    """
    通过AWS CLI命令下载指定年份的NOAA GSOD数据
    
    参数:
    year (str): 要下载的年份
    
    返回:
    bool: 下载是否成功
    """
    # 构造本地目录路径
    base_dir = "./data/noaa-gsod-pds"
    year_dir = f"{base_dir}/{year}"
    
    # 确保目录存在
    os.makedirs(year_dir, exist_ok=True)
    
    # 构造AWS S3 ls命令，列出该年份的所有文件
    ls_command = f"aws s3 ls --no-sign-request s3://noaa-gsod-pds/{year}/ --recursive"
    
    try:
        # 执行ls命令获取文件列表
        logger.info(f"获取S3中{year}年的文件列表...")
        result = subprocess.run(ls_command, shell=True, check=True, capture_output=True, text=True)
        file_list = result.stdout.strip().split('\n')
        
        # 解析文件列表，提取文件名
        s3_files = []
        for line in file_list:
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    file_path = parts[3]  # S3路径格式：YYYY/stationid.csv
                    if file_path.endswith('.csv'):
                        s3_files.append(file_path)
        
        logger.info(f"找到{len(s3_files)}个文件需要下载")
        
        # 下载每个文件，除非本地已存在
        downloaded_count = 0
        skipped_count = 0
        
        for s3_file in s3_files:
            # 提取站点ID
            station_id = os.path.basename(s3_file)
            local_file = f"{year_dir}/{station_id}"
            
            # 检查本地文件是否存在
            if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                logger.debug(f"跳过已存在的文件: {local_file}")
                skipped_count += 1
                continue
            
            # 构造下载命令
            download_command = f"aws s3 cp --no-sign-request s3://noaa-gsod-pds/{s3_file} {local_file}"
            
            # 执行下载
            try:
                subprocess.run(download_command, shell=True, check=True)
                downloaded_count += 1
                logger.debug(f"已下载: {local_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"下载文件失败 {s3_file}: {str(e)}")
        
        logger.info(f"下载完成: {downloaded_count}个文件已下载, {skipped_count}个文件已跳过")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"执行AWS CLI命令失败: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"下载数据时出错: {str(e)}")
        return False

def get_noaa_location_ids(years):
    """
    获取指定年份的所有STATION
    """
    location_ids = []
    # 构造AWS S3 ls命令，列出该年份的所有文件
    for year in years:
        ls_command = f"aws s3 ls --no-sign-request s3://noaa-gsod-pds/{year}/ --recursive"
        result = subprocess.run(ls_command, shell=True, check=True, capture_output=True, text=True)
        file_list = result.stdout.strip().split('\n')
            
        # 解析文件列表，提取文件名
        for line in file_list:
            if line.strip():
                parts = line.split()
                print(parts)
                if len(parts) >= 4:
                    file_path = parts[3]  # S3路径格式：YYYY/stationid.csv
                    if file_path.endswith('.csv'):
                        location_id = file_path.split("/")[1].strip('.csv')
                        print(location_id)
                        location_ids.append(location_id)

    return location_ids


def get_noaa_locations_info(file_path, stations_file="data/noaa-gsod-pds/stations_file.csv"):
    """
    从指定目录下的NOAA数据文件中提取站点信息
    
    参数:
    file_path (str): 包含NOAA数据文件的目录路径
    stations_file (str): 保存站点信息的CSV文件路径
    
    返回:
    pandas.DataFrame: 包含唯一站点信息的DataFrame
    """
    # 需要提取的列
    required_columns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME"]
    
    # 初始化DataFrame
    if os.path.exists(stations_file):
        logger.info(f"从{stations_file}加载现有站点信息")
        df = pd.read_csv(stations_file)
        # 确保所有必要的列都存在
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
    else:
        logger.info("创建新的站点信息DataFrame")
        df = pd.DataFrame(columns=required_columns)
    
    # 获取现有站点ID集合，用于快速检查重复
    existing_stations = set(df['STATION'].dropna().unique())
    logger.info(f"现有站点数量: {len(existing_stations)}")
    
    # 如果file_path是目录，则遍历所有文件
    if os.path.isdir(file_path):
        files = []
        # 递归遍历目录
        for root, _, filenames in os.walk(file_path):
            for filename in filenames:
                if filename.endswith('.csv'):
                    files.append(os.path.join(root, filename))
    else:
        # 如果是单个文件，则只处理该文件
        files = [file_path] if file_path.endswith('.csv') else []
    
    logger.info(f"找到{len(files)}个CSV文件需要处理")
    
    # 处理每个文件
    new_stations_count = 0
    for file in files:
        try:
            # 只读取第一行数据
            temp_df = pd.read_csv(file, nrows=1)
            
            # 检查是否包含所需列
            missing_cols = [col for col in required_columns if col not in temp_df.columns]
            if missing_cols:
                logger.warning(f"文件{file}缺少列: {missing_cols}")
                continue
                
            # 提取站点ID
            station_id = temp_df['STATION'].iloc[0]
            
            # 如果站点已存在，则跳过
            if station_id in existing_stations:
                continue
                
            # 提取所需列
            station_info = temp_df[required_columns].iloc[0:1]
            
            # 添加到主DataFrame
            df = pd.concat([df, station_info], ignore_index=True)
            existing_stations.add(station_id)
            new_stations_count += 1
            
            if new_stations_count % 100 == 0:
                logger.info(f"已处理{new_stations_count}个新站点")
                
        except Exception as e:
            logger.error(f"处理文件{file}时出错: {str(e)}")
    
    logger.info(f"共添加了{new_stations_count}个新站点，总站点数: {len(df)}")
    
    # 保存到CSV文件
    os.makedirs(os.path.dirname(os.path.abspath(stations_file)), exist_ok=True)
    df.to_csv(stations_file, index=False)
    logger.info(f"站点信息已保存到: {stations_file}")
    
    return df


def get_openaq_stations(stations_file="data/noaa-gsod-pds/stations_file.csv", station_location_file="data/noaa-openaq-mapping.csv", api_key=None, radius_m=16100):
    """
    将NOAA站点与OpenAQ监测站点进行映射
    
    参数:
    stations_file (str): NOAA站点信息文件路径
    station_location_file (str): 输出的站点映射文件路径
    api_key (str, optional): OpenAQ API密钥
    radius_m (int): 搜索半径（米）
    
    返回:
    pandas.DataFrame: 包含NOAA站点与OpenAQ监测站点映射的DataFrame
    """
    # 检查stations_file是否存在
    if not os.path.exists(stations_file):
        logger.error(f"站点文件不存在: {stations_file}")
        return pd.DataFrame()
    
    # 读取NOAA站点数据
    logger.info(f"从{stations_file}读取NOAA站点数据")
    try:
        noaa_stations = pd.read_csv(stations_file)
    except Exception as e:
        logger.error(f"读取站点文件时出错: {str(e)}")
        return pd.DataFrame()
    
    # 检查必要的列是否存在
    required_columns = ["STATION", "LATITUDE", "LONGITUDE"]
    missing_columns = [col for col in required_columns if col not in noaa_stations.columns]
    if missing_columns:
        logger.error(f"站点文件缺少必要的列: {missing_columns}")
        return pd.DataFrame()
    
    # 初始化结果DataFrame
    result_data = []
    
    # 处理每个NOAA站点
    total_stations = len(noaa_stations)
    logger.info(f"开始处理{total_stations}个NOAA站点")
    
    for idx, station in noaa_stations.iterrows():
        station_id = station["STATION"]
        lat = station["LATITUDE"]
        lon = station["LONGITUDE"]
        
        # 跳过无效的坐标
        if pd.isna(lat) or pd.isna(lon):
            logger.warning(f"站点{station_id}的坐标无效，跳过")
            continue
        
        logger.info(f"处理NOAA站点 {idx+1}/{total_stations}: {station_id} ({lat}, {lon})")
        
        try:
            # 获取OpenAQ监测站点IDs
            openaq_ids = fetch_openaq_location_ids(lat, lon, api_key=api_key, radius_m=radius_m)
            
            if not openaq_ids:
                logger.info(f"站点{station_id}周围{radius_m}米内未找到OpenAQ监测站点")
                continue
            
            logger.info(f"站点{station_id}周围找到{len(openaq_ids)}个OpenAQ监测站点")
            
            # 为每个OpenAQ ID创建一条记录
            for openaq_id in openaq_ids:
                # 获取OpenAQ站点详细信息
                openaq_location = get_location_by_id(openaq_id, api_key=api_key)
                
                if not openaq_location:
                    logger.warning(f"无法获取OpenAQ站点{openaq_id}的详细信息")
                    continue
                
                # 创建包含NOAA站点和OpenAQ站点信息的记录
                record = station.to_dict()
                record["OPENAQ_ID"] = openaq_id
                record["OPENAQ_NAME"] = openaq_location.get("name", "")
                record["OPENAQ_CITY"] = openaq_location.get("city", "")
                record["OPENAQ_COUNTRY"] = openaq_location.get("country", "")
                record["OPENAQ_LATITUDE"] = openaq_location.get("coordinates", {}).get("latitude", "")
                record["OPENAQ_LONGITUDE"] = openaq_location.get("coordinates", {}).get("longitude", "")
                
                # 添加监测的参数
                parameters = openaq_location.get("parameters", [])
                record["OPENAQ_PARAMETERS"] = ", ".join([p.get("parameter", "") for p in parameters])
                
                result_data.append(record)
                
        except Exception as e:
            logger.error(f"处理站点{station_id}时出错: {str(e)}")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(result_data)
    
    # 保存结果
    if not result_df.empty:
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(os.path.abspath(station_location_file)), exist_ok=True)
            result_df.to_csv(station_location_file, index=False)
            logger.info(f"已将{len(result_df)}条映射记录保存到{station_location_file}")
        except Exception as e:
            logger.error(f"保存映射文件时出错: {str(e)}")
    else:
        logger.warning("未找到任何映射关系，不保存文件")
    
    return result_df


def load_noaa_location_data(file_path):
    """
    加载NOAA位置数据
    
    参数:
    file_path (str): 数据文件路径
    
    返回:
    pandas.DataFrame: 加载的位置数据
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"从 {file_path} 加载了 {len(df)} 个站点位置记录")
            return df
        else:
            logger.info(f"文件 {file_path} 不存在，将创建新的位置数据")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"加载位置数据时出错: {e}")
        return pd.DataFrame()


def get_stations_info_from_df(df, stations_file="data/noaa-gsod-pds/stations_file.csv"):
    """
    从DataFrame中提取站点信息并保存到文件
    
    参数:
    df (pandas.DataFrame): 包含NOAA数据的DataFrame
    stations_file (str): 保存站点信息的CSV文件路径
    
    返回:
    pandas.DataFrame: 包含唯一站点信息的DataFrame
    """
    # 需要提取的列
    required_columns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME"]
    
    # 检查必要的列是否存在
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"输入DataFrame缺少列: {missing_columns}")
        # 为缺失的列添加空值
        for col in missing_columns:
            df[col] = None
    
    # 初始化结果DataFrame
    if os.path.exists(stations_file):
        logger.info(f"从{stations_file}加载现有站点信息")
        stations_df = pd.read_csv(stations_file)
        # 确保所有必要的列都存在
        for col in required_columns:
            if col not in stations_df.columns:
                stations_df[col] = None
    else:
        logger.info("创建新的站点信息DataFrame")
        stations_df = pd.DataFrame(columns=required_columns)
    
    # 获取现有站点ID集合，用于快速检查重复
    existing_stations = set(stations_df['STATION'].dropna().unique())
    logger.info(f"现有站点数量: {len(existing_stations)}")
    
    # 获取输入DataFrame中的唯一站点
    unique_stations = df['STATION'].dropna().unique()
    logger.info(f"输入数据中有{len(unique_stations)}个唯一站点")
    
    # 处理每个唯一站点
    new_stations_count = 0
    for station_id in unique_stations:
        # 如果站点已存在，则跳过
        if station_id in existing_stations:
            continue
            
        # 获取该站点的第一条记录
        station_data = df[df['STATION'] == station_id].iloc[0:1]
        
        # 提取所需列
        station_info = station_data[required_columns]
        
        # 添加到主DataFrame
        stations_df = pd.concat([stations_df, station_info], ignore_index=True)
        existing_stations.add(station_id)
        new_stations_count += 1
        
        if new_stations_count % 100 == 0:
            logger.info(f"已处理{new_stations_count}个新站点")
    
    logger.info(f"共添加了{new_stations_count}个新站点，总站点数: {len(stations_df)}")
    
    # 保存到CSV文件
    os.makedirs(os.path.dirname(os.path.abspath(stations_file)), exist_ok=True)
    stations_df.to_csv(stations_file, index=False)
    logger.info(f"站点信息已保存到: {stations_file}")
    
    return stations_df


if __name__ == "__main__":
#    print(download_nooa_from_s3("2024"))
#    print(get_noaa_locations_info("./data/noaa-gsod-pds/2024"))
#    get_openaq_stations(api_key="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00")
    print(len(get_noaa_location_ids([2025])))