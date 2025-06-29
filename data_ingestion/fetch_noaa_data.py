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
from typing import List, Dict, Optional, Union, Any, Tuple, Set
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from openaq_utils import fetch_openaq_location_ids, get_location_by_id
from utils import *

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NOAADataFetcher:
    """NOAA 气象数据获取与处理类"""
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化 NOAA 数据获取器
        
        参数:
        data_dir (str, optional): 数据存储目录，默认为 './data/noaa-gsod-pds'
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), "data/noaa-gsod-pds")
        else:
            self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # S3 相关配置
        self.bucket_name = 'noaa-gsod-pds'
        self.s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    def get_noaa_data_frame(self, stations: List[str], start_date: str, end_date: str) -> pd.DataFrame:
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
                df = self.get_noaa_data_frame_by_id(station_id, year_str)
                
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
    
    def get_noaa_data_frame_by_id(self, station_id: str, year: str) -> pd.DataFrame:
        """
        根据指定的站点ID和年份获取NOAA GSOD数据
        ref: https://github.com/aws-samples/aws-smsl-predict-airquality-via-weather/
        
        参数:
        station_id (str): NOAA站点ID
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
        file_dir = os.path.join(self.data_dir, year)
        file_path = os.path.join(file_dir, f"{station_id}.csv")
        
        # 确保目录存在
        os.makedirs(file_dir, exist_ok=True)
        
        # 检查是否已经存在日期特定的数据文件
        if os.path.exists(file_path):
            logger.info(f"从本地文件加载数据: {file_path}")
            return pd.read_csv(file_path)
        else:       
            # 如果本地没有数据，从S3下载
            logger.info(f"从S3下载数据 (bucket: {self.bucket_name})...")
            
            try:
                key = f'{year}/{station_id}.csv'
                
                # 获取S3对象
                csv_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                csv_string = csv_obj['Body'].read().decode('utf-8')
                
                df = pd.read_csv(StringIO(csv_string))
                df.to_csv(file_path, index=False)
                logger.info(f"已保存数据到: {file_path}")
                
                return df
                            
            except Exception as e:
                logger.error(f"下载或处理数据时出错: {str(e)}")
                return pd.DataFrame()
    
    def download_noaa_from_s3(self, year: str) -> bool:
        """
        通过AWS CLI命令下载指定年份的NOAA GSOD数据
        
        参数:
        year (str): 要下载的年份
        
        返回:
        bool: 下载是否成功
        """
        # 构造本地目录路径
        year_dir = os.path.join(self.data_dir, year)
        
        # 确保目录存在
        os.makedirs(year_dir, exist_ok=True)
        
        # 构造AWS S3 ls命令，列出该年份的所有文件
        ls_command = f"aws s3 ls --no-sign-request s3://{self.bucket_name}/{year}/ --recursive"
        
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
                local_file = os.path.join(year_dir, station_id)
                
                # 检查本地文件是否存在
                if os.path.exists(local_file) and os.path.getsize(local_file) > 0:
                    logger.debug(f"跳过已存在的文件: {local_file}")
                    skipped_count += 1
                    continue
                
                # 构造下载命令
                download_command = f"aws s3 cp --no-sign-request s3://{self.bucket_name}/{s3_file} {local_file}"
                
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
    
    def get_noaa_location_ids(self, years: List[int]) -> List[str]:
        """
        获取指定年份的所有STATION
        
        参数:
        years (List[int]): 年份列表
        
        返回:
        List[str]: 站点ID列表
        """
        location_ids = []
        # 构造AWS S3 ls命令，列出该年份的所有文件
        for year in years:
            ls_command = f"aws s3 ls --no-sign-request s3://{self.bucket_name}/{year}/ --recursive"
            try:
                result = subprocess.run(ls_command, shell=True, check=True, capture_output=True, text=True)
                file_list = result.stdout.strip().split('\n')
                    
                # 解析文件列表，提取文件名
                for line in file_list:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 4:
                            file_path = parts[3]  # S3路径格式：YYYY/stationid.csv
                            if file_path.endswith('.csv'):
                                location_id = file_path.split("/")[1].strip('.csv')
                                location_ids.append(location_id)
            except Exception as e:
                logger.error(f"获取{year}年站点ID时出错: {str(e)}")
        
        # 去重
        unique_location_ids = list(set(location_ids))
        logger.info(f"共找到{len(unique_location_ids)}个唯一站点ID")
        return unique_location_ids
    
    def get_openaq_stations(self, 
                           stations_file: str = "data/noaa-gsod-pds/stations_file.csv", 
                           station_location_file: str = "data/noaa-openaq-mapping.csv", 
                           api_key: Optional[str] = None, 
                           radius_m: int = 16100) -> pd.DataFrame:
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
    
    def get_stations_info_from_df(self, df: pd.DataFrame, stations_file: str = "data/noaa-gsod-pds/stations_file.csv") -> pd.DataFrame:
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


def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='获取和处理NOAA气象数据')
    parser.add_argument('--data_dir', type=str, help='数据存储目录，默认为./data/noaa-gsod-pds')
    parser.add_argument('--download_year', type=str, help='下载指定年份的所有数据')
    parser.add_argument('--station_id', type=str, help='获取指定站点的数据')
    parser.add_argument('--year', type=str, help='获取指定年份的数据')
    parser.add_argument('--start_date', type=str, help='开始日期，格式为YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYY-MM-DD')
    parser.add_argument('--stations_file', type=str, default='data/noaa-gsod-pds/stations_file.csv', 
                        help='NOAA站点信息文件路径')
    parser.add_argument('--mapping_file', type=str, default='data/noaa-openaq-mapping.csv', 
                        help='NOAA-OpenAQ站点映射文件路径')
    parser.add_argument('--api_key', type=str, help='OpenAQ API密钥')
    parser.add_argument('--radius', type=int, default=16100, help='搜索OpenAQ站点的半径（米）')
    parser.add_argument('--list_stations', action='store_true', help='列出指定年份的所有站点')
    parser.add_argument('--map_stations', action='store_true', help='将NOAA站点与OpenAQ站点进行映射')
    
    args = parser.parse_args()
    
    # 创建NOAA数据获取器
    fetcher = NOAADataFetcher(args.data_dir)
    
    # 下载指定年份的数据
    if args.download_year:
        logger.info(f"下载{args.download_year}年的NOAA数据")
        success = fetcher.download_noaa_from_s3(args.download_year)
        if success:
            logger.info(f"成功下载{args.download_year}年的数据")
        else:
            logger.error(f"下载{args.download_year}年的数据失败")
    
    # 获取指定站点和年份的数据
    elif args.station_id and args.year:
        logger.info(f"获取站点{args.station_id}在{args.year}年的数据")
        df = fetcher.get_noaa_data_frame_by_id(args.station_id, args.year)
        if not df.empty:
            logger.info(f"获取到{len(df)}条记录")
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"数据样例:\n{df.head()}")
        else:
            logger.warning(f"未获取到数据")
    
    # 获取指定站点在日期范围内的数据
    elif args.station_id and args.start_date and args.end_date:
        logger.info(f"获取站点{args.station_id}从{args.start_date}到{args.end_date}的数据")
        df = fetcher.get_noaa_data_frame([args.station_id], args.start_date, args.end_date)
        if not df.empty:
            logger.info(f"获取到{len(df)}条记录")
            logger.info(f"数据列: {df.columns.tolist()}")
            logger.info(f"数据样例:\n{df.head()}")
        else:
            logger.warning(f"未获取到数据")
    
    # 列出指定年份的所有站点
    elif args.list_stations and args.year:
        logger.info(f"列出{args.year}年的所有站点")
        location_ids = fetcher.get_noaa_location_ids([int(args.year)])
        logger.info(f"共找到{len(location_ids)}个站点")
        for i, station_id in enumerate(location_ids[:10]):
            logger.info(f"站点 {i+1}: {station_id}")
        if len(location_ids) > 10:
            logger.info(f"... 以及其他 {len(location_ids)-10} 个站点")
    
    # 将NOAA站点与OpenAQ站点进行映射
    elif args.map_stations:
        logger.info("将NOAA站点与OpenAQ站点进行映射")
        result_df = fetcher.get_openaq_stations(
            stations_file=args.stations_file,
            station_location_file=args.mapping_file,
            api_key=args.api_key,
            radius_m=args.radius
        )
        if not result_df.empty:
            logger.info(f"映射了{len(result_df)}个站点")
            logger.info(f"映射结果样例:\n{result_df.head()}")
        else:
            logger.warning("未找到任何映射关系")
    
    else:
        logger.warning("未指定操作，请使用--help查看帮助")


if __name__ == "__main__":
    main()