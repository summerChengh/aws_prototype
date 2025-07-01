#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入所需的库和函数
import os
import pandas as pd
import random
import sys
import logging
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fetch_noaa_data import NOAADataFetcher
from utils import get_years_from_time_range
from fetch_openaq_data import OpenAQDataFetcher
from cal_aqi import AQICalculator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AirQualityDataProcessor:
    """空气质量数据处理器，用于处理NOAA和OpenAQ数据并合并"""

    def __init__(self, start_date: str, end_date: str, output_dir: str = None, api_key: str = None):
        """
        初始化数据处理器
        
        参数:
        start_date (str): 开始日期，格式为'YYYY-MM-DD'
        end_date (str): 结束日期，格式为'YYYY-MM-DD'
        output_dir (str, optional): 输出目录，默认为'./data/processed/{start_date}_{end_date}'
        api_key (str, optional): OpenAQ API密钥
        """
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_openaq = start_date.replace("-", "")  # 转换为'YYYYMMDD'格式
        self.end_date_openaq = end_date.replace("-", "")  # 转换为'YYYYMMDD'格式

        # 设置输出目录
        if output_dir is None:
            self.output_dir = f"./data/processed/{self.start_date_openaq}_{self.end_date_openaq}"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # API密钥
        self.api_key = api_key

        # 数据存储
        self.nooa_df = None
        self.mapping_df = None
        self.merged_df = None
        self.final_merged = None

        # 创建数据获取器
        self.noaa_fetcher = NOAADataFetcher()
        self.openaq_fetcher = OpenAQDataFetcher()

        # AQI计算器
        self.calculator = AQICalculator()

        # 污染物映射
        self.pollutants = {
            'pm25_24h': 'PM2.5_24h',
            'pm10_24h': 'PM10_24h',
            'o3_8h': 'O3_8h',
            'o3_1h': 'O3_1h',
            'co_8h': 'CO_8h',
            'so2_1h': 'SO2_1h',
            'so2_24h': 'SO2_24h',
            'no2_1h': 'NO2_1h'
        }

    def get_noaa_data(self, station_ids: List[str] = None, sample_rate: float = 0.001) -> pd.DataFrame:
        """
        获取NOAA气象数据
        
        参数:
        station_ids (List[str], optional): 指定的气象站ID列表
        sample_rate (float, optional): 如果未提供station_ids，则按此比例随机采样
        
        返回:
        pd.DataFrame: NOAA气象数据
        """
        nooa_file = f"{self.output_dir}/nooa.csv"

        # 如果已有缓存文件，直接加载
        if os.path.exists(nooa_file):
            logger.info(f"NOAA数据文件已存在，已加载: {nooa_file}")
            self.nooa_df = pd.read_csv(nooa_file)
            return self.nooa_df

        # 如果未提供station_ids，则获取所有站点并采样
        if station_ids is None:
            years = get_years_from_time_range(self.start_date, self.end_date)
            all_station_ids = self.noaa_fetcher.get_noaa_location_ids(years)

            # 确保采样数量至少为1
            sample_size = max(1, int(len(all_station_ids) * sample_rate))
            station_ids = random.sample(all_station_ids, sample_size)
            logger.info(f"采样了 {sample_size} 个站点，从总共 {len(all_station_ids)} 个站点中")

        # 获取NOAA数据
        self.nooa_df = self.noaa_fetcher.get_noaa_data_frame(station_ids, self.start_date, self.end_date)

        # 保存数据
        self.nooa_df.to_csv(nooa_file, index=False)
        logger.info(f"NOAA数据已保存到: {nooa_file}")

        return self.nooa_df

    def generate_station_info(self) -> str:
        """
        从NOAA数据中提取站点信息
        
        返回:
        str: 站点信息文件路径
        """
        if self.nooa_df is None:
            raise ValueError("请先调用get_noaa_data获取NOAA数据")

        stations_file = f"{self.output_dir}/nooa_stations_file.csv"

        if not os.path.exists(stations_file):
            self.noaa_fetcher.get_stations_info_from_df(self.nooa_df, stations_file)
            logger.info(f"已保存站点信息到: {stations_file}")
        else:
            logger.info(f"站点信息文件已存在: {stations_file}")

        return stations_file

    def map_noaa_to_openaq(self, stations_file: str = None) -> pd.DataFrame:
        """
        将NOAA站点映射到最近的OpenAQ站点
        
        参数:
        stations_file (str, optional): 站点信息文件路径，如果未提供则调用generate_station_info
        
        返回:
        pd.DataFrame: 映射关系DataFrame
        """
        if stations_file is None:
            stations_file = self.generate_station_info()

        station_location_file = f"{self.output_dir}/noaa-openaq-mapping.csv"

        if not os.path.exists(station_location_file):
            self.noaa_fetcher.get_openaq_stations(stations_file, station_location_file, api_key=self.api_key,
                                                  radius_m=8050)
            logger.info(f"已生成NOAA到OpenAQ的映射关系: {station_location_file}")
        else:
            logger.info(f"映射关系文件已存在: {station_location_file}")

        # 加载映射关系
        self.mapping_df = pd.read_csv(station_location_file)

        return self.mapping_df

    def process_openaq_data(self) -> pd.DataFrame:
        """
        处理OpenAQ数据并计算AQI
        
        返回:
        pd.DataFrame: 处理后的OpenAQ数据，包含AQI
        """
        if self.mapping_df is None:
            self.map_noaa_to_openaq()

        # 获取所有不重复的OpenAQ ID
        unique_openaq_ids = self.mapping_df['OPENAQ_ID'].dropna().unique()
        logger.info(f"共找到 {len(unique_openaq_ids)} 个不重复的OPENAQ_ID")

        # 合并所有ID的processed_df到一张表
        all_processed_dfs = []

        for id in unique_openaq_ids:
            merged_aq_df, _, processed_df = self.openaq_fetcher.process_aq_data_and_extract_features(
                id,
                output_dir=f"{self.output_dir}/{id}",
                start_date=self.start_date_openaq,
                end_date=self.end_date_openaq,
                download_missing=True
            )

            if processed_df is not None and not processed_df.empty:
                # 增加ID列以便区分
                processed_df['OPENAQ_ID'] = id
                all_processed_dfs.append(processed_df)

        if all_processed_dfs:
            self.merged_df = pd.concat(all_processed_dfs, ignore_index=True)
            merged_file = f"{self.output_dir}/all_openaq_processed.csv"
            self.merged_df.to_csv(merged_file, index=False)
            logger.info(f"所有ID的processed_df已合并并保存到: {merged_file}")
        else:
            logger.warning("没有可合并的processed_df，未生成合并文件")
            return None

        # 计算AQI
        self._calculate_aqi()

        return self.merged_df

    def _calculate_aqi(self) -> None:
        """计算AQI并添加到merged_df中"""
        if self.merged_df is None or self.merged_df.empty:
            logger.warning("没有数据可计算AQI")
            return

        # 应用函数到每一行
        aqi_results = self.merged_df.apply(self._calculate_row_aqi, axis=1)

        # 提取综合AQI和主要污染物
        self.merged_df['AQI'] = [result[0] if result else None for result in aqi_results]
        self.merged_df['main_pollutant'] = [result[1] if result else None for result in aqi_results]

        # 添加AQI类别
        self.merged_df['AQI_category'] = self.merged_df['AQI'].apply(
            lambda x: self.calculator.get_aqi_category(x)[0] if pd.notnull(x) else None
        )

        # 提取单个污染物的AQI值
        for col in ["O3", "SO2", "PM2.5_24h", "PM10_24h", "CO_8h", "NO2_1h"]:
            aqi_col = f"{col}_aqi"
            self.merged_df[aqi_col] = [result[2][col][1] if result and col in result[2] else None for result in
                                       aqi_results]
            logger.info(f"已从calculate_overall_aqi结果中提取{col}的AQI并保存到{aqi_col}列")

        # 保存带有AQI的数据
        aqi_file = f"{self.output_dir}/all_openaq_with_aqi.csv"
        self.merged_df.to_csv(aqi_file, index=False)
        logger.info(f"已计算AQI并保存到: {aqi_file}")

    def _calculate_row_aqi(self, row: pd.Series) -> Tuple[Optional[float], Optional[str], Dict]:
        """
        计算单行数据的AQI
        
        参数:
        row (pd.Series): 单行数据
        
        返回:
        Tuple[Optional[float], Optional[str], Dict]: (AQI值, 主要污染物, 单个污染物AQI)
        """
        # 构建浓度字典
        concentrations = {}
        for col, pollutant_type in self.pollutants.items():
            if col in row.index and pd.notnull(row[col]):
                concentrations[pollutant_type] = row[col]
    
        # 如果没有有效的污染物数据，返回None
        if not concentrations:
            return None, None, {}
    
        # 计算综合AQI和单个污染物AQI
        overall_aqi, main_pollutant, single_aqis = self.calculator.calculate_overall_aqi(concentrations)
        return overall_aqi, main_pollutant, single_aqis

    def merge_noaa_openaq_data(self) -> pd.DataFrame:
        """
        合并NOAA和OpenAQ数据
        
        返回:
        pd.DataFrame: 合并后的数据
        """
        if self.nooa_df is None:
            logger.warning("NOAA数据未加载，请先调用get_noaa_data")
            return None

        if self.merged_df is None:
            logger.warning("OpenAQ数据未处理，请先调用process_openaq_data")
            return None

        # 确保字段名一致
        nooa_df_merged = self.nooa_df.copy()
        if 'STATION' not in nooa_df_merged.columns:
            # 有些NOAA数据可能用'station'小写
            if 'station' in nooa_df_merged.columns:
                nooa_df_merged.rename(columns={'station': 'STATION'}, inplace=True)

        # 将DATE列重命名为date
        if 'DATE' in nooa_df_merged.columns:
            nooa_df_merged.rename(columns={'DATE': 'date'}, inplace=True)

        # 只保留mapping_df中有用的两列
        mapping_bridge = self.mapping_df[['STATION', 'OPENAQ_ID']].dropna()

        # 先将NOAA数据和mapping_df左连接，获得OPENAQ_ID
        nooa_with_openaq = pd.merge(nooa_df_merged, mapping_bridge, on='STATION', how='left')

        # 确保两个表的 date 列格式一致
        nooa_with_openaq['date'] = pd.to_datetime(nooa_with_openaq['date']).dt.strftime('%Y-%m-%d')
        self.merged_df['date'] = pd.to_datetime(self.merged_df['date']).dt.strftime('%Y-%m-%d')

        # 然后再进行合并
        self.final_merged = pd.merge(nooa_with_openaq, self.merged_df,
                                     on=['OPENAQ_ID', 'date'],
                                     how='inner',
                                     suffixes=('_noaa', '_openaq'))

        # 保存最终合并结果
        final_merged_file = f"{self.output_dir}/nooa_openaq_merged.csv"
        self.final_merged.to_csv(final_merged_file, index=False)
        logger.info(f"NOAA与OpenAQ数据已左连接并保存到: {final_merged_file}")

        return self.final_merged

    def run_full_pipeline(self, station_ids: List[str] = None) -> pd.DataFrame:
        """
        运行完整的数据处理流程
        
        参数:
        station_ids (List[str], optional): 指定的气象站ID列表
        
        返回:
        pd.DataFrame: 最终合并的数据
        """
        # 1. 获取NOAA数据
        self.get_noaa_data(station_ids)

        # 2. 生成站点信息
        stations_file = self.generate_station_info()

        # 3. 映射NOAA到OpenAQ
        self.map_noaa_to_openaq(stations_file)

        # 4. 处理OpenAQ数据并计算AQI
        self.process_openaq_data()

        # 5. 合并NOAA和OpenAQ数据
        final_data = self.merge_noaa_openaq_data()

        logger.info("数据处理完成")
        return final_data

    @staticmethod
    def cal_window_feature(df, cols, window_size):
        """
        计算过去n天平均值特征
        """
        for col in cols:
            df[f'{col}_mean_{window_size}d'] = (
                df[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
        return df


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description='处理NOAA和OpenAQ数据并合并')
    parser.add_argument('--start_date', type=str, required=True, help='开始日期，格式为YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, required=True, help='结束日期，格式为YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--api_key', type=str, help='OpenAQ API密钥')
    parser.add_argument('--stations', type=str, nargs='+', help='指定的NOAA站点ID列表')

    args = parser.parse_args()

    # 创建处理器并运行
    processor = AirQualityDataProcessor(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        api_key=args.api_key
    )

    processor.run_full_pipeline(args.stations)


if __name__ == '__main__':
    api_key = os.getenv("OPENAQ_API_KEY")
    if not api_key:
        raise RuntimeError("API key not set in environment variable OPENAQ_API_KEY")
    # 示例用法
    # 如果直接运行此脚本，使用默认参数
    start_date = "2016-01-01"
    end_date = "2025-06-29"

    # 预定义的站点ID列表
    sampled_stations = ["72384023155", "72389093193", "72389693144", "72494693232",
                        "72287493134", "72278023183", "70261026411", "41640099999"]

    processor = AirQualityDataProcessor(start_date, end_date,
                                        api_key=api_key)
    processor.run_full_pipeline(sampled_stations)
