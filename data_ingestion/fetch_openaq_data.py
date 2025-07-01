#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import argparse
import sys
import subprocess
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Union, Any, Tuple, Set

# 导入工具函数
from utils import load_csv_file, find_data_files, preprocess_datetime_column, sort_by_datetime, \
    load_and_merge_data_files, perform_time_resampling, get_years_from_time_range

# 导入OpenAQProcessor类
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_ingestion.openaq_feats import OpenAQProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAQDataFetcher:
    """OpenAQ 数据获取与处理类"""

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化 OpenAQ 数据获取器
        
        参数:
        data_dir (str, optional): 数据存储目录，默认为 './data/opendb-aq'
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.getcwd(), "data/opendb-aq")
        else:
            self.data_dir = data_dir

        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 初始化缓存，用于记录已知不存在的S3路径
        self._nonexistent_s3_paths = set()

        # 缓存文件路径
        self._cache_file = os.path.join(self.data_dir, "nonexistent_s3_paths.json")

        # 加载缓存
        self._load_nonexistent_paths_cache()

    def _load_nonexistent_paths_cache(self):
        """加载不存在S3路径的缓存"""
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self._nonexistent_s3_paths = set(cache_data.get('paths', []))
                    logger.info(f"已加载 {len(self._nonexistent_s3_paths)} 条不存在的S3路径缓存")
            except Exception as e:
                logger.warning(f"加载S3路径缓存失败: {e}")
                self._nonexistent_s3_paths = set()

    def _save_nonexistent_paths_cache(self):
        """保存不存在S3路径的缓存"""
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            with open(self._cache_file, 'w') as f:
                json.dump({
                    'paths': list(self._nonexistent_s3_paths),
                    'updated_at': datetime.now().isoformat()
                }, f)
            logger.info(f"已保存 {len(self._nonexistent_s3_paths)} 条不存在的S3路径缓存")
        except Exception as e:
            logger.warning(f"保存S3路径缓存失败: {e}")

    def download_data_from_s3(self, location_id: str, start_date: str, end_date: str) -> bool:
        """
        从AWS S3下载指定日期范围内的数据文件
        只下载指定日期范围内的文件，而不是整个月份的数据
        
        参数:
        location_id (str): 位置ID
        start_date (str): 起始日期，格式为YYYYMMDD
        end_date (str): 结束日期，格式为YYYYMMDD
        
        返回:
        bool: 是否成功下载数据
        """
        if not start_date or not end_date:
            logger.error("下载数据需要指定起始日期和结束日期")
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
            skipped_nonexistent_count = 0

            # 对每个年月组合执行下载
            for (year, month), days in sorted(dates_by_year_month.items()):
                year_str = str(year)
                month_str = f"{month:02d}"

                # 确保本地目录存在
                local_dir = os.path.join(self.data_dir, f"locationid={location_id}/year={year_str}/month={month_str}")
                os.makedirs(local_dir, exist_ok=True)

                # 对每一天分别下载
                for day in sorted(days):
                    day_str = f"{day:02d}"
                    file_name = f"location-{location_id}-{year_str}{month_str}{day_str}.csv.gz"

                    # 检查本地文件是否已存在
                    local_file = os.path.join(local_dir, file_name)
                    if os.path.exists(local_file):
                        logger.info(f"文件已存在，跳过下载: {file_name}")
                        downloaded_files.append(local_file)
                        download_success = True
                        continue

                    # 构建S3路径和本地路径
                    s3_file = f"s3://openaq-data-archive/records/csv.gz/locationid={location_id}/year={year_str}/month={month_str}/{file_name}"

                    # 检查是否在已知不存在的S3路径缓存中
                    if file_name in self._nonexistent_s3_paths:
                        logger.debug(f"根据缓存，S3文件不存在，跳过: {s3_file}")
                        continue

                    # 检查S3文件是否存在
                    check_cmd = [
                        "aws", "s3", "ls",
                        "--no-sign-request",
                        s3_file
                    ]

                    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
                    if check_result.returncode != 0 or not check_result.stdout.strip():
                        logger.warning(f"S3文件不存在，跳过: {s3_file}")
                        # 添加到不存在的S3路径缓存中
                        self._nonexistent_s3_paths.add(file_name)
                        skipped_nonexistent_count += 1
                        continue

                    logger.info(f"正在从S3下载文件: {s3_file} -> {local_file}")

                    # 执行AWS CLI命令下载单个文件
                    cmd = [
                        "aws", "s3", "cp",
                        "--no-sign-request",
                        s3_file,
                        local_file
                    ]

                    try:
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        logger.info(f"下载完成: {file_name}")
                        downloaded_files.append(local_file)
                        if file_name in self._nonexistent_s3_paths:
                            self._nonexistent_s3_paths.remove(file_name)
                        download_success = True
                    except subprocess.CalledProcessError as e:
                        logger.error(f"下载失败: {file_name}")
                        logger.error(f"错误输出: {e.stderr}")
                        # 如果下载失败，可能是因为文件不存在，添加到缓存
                        self._nonexistent_s3_paths.add(file_name)

            logger.info(
                f"总共下载了 {len(downloaded_files)} 个文件，跳过了 {skipped_nonexistent_count} 个已知不存在的文件")

            # 保存更新后的缓存
            if skipped_nonexistent_count > 0 or len(downloaded_files) > 0:
                self._save_nonexistent_paths_cache()

            return download_success

        except Exception as e:
            logger.error(f"下载数据时出错: {e}")
            return False

    def fetch_data_from_s3_by_year(self, location_id: str, year: int, check_existing: bool = True) -> Tuple[bool, int]:
        """
        从AWS S3按年下载数据文件，使用递归方式一次性下载整年数据
        
        参数:
        location_id (str): 位置ID
        year (int): 年份，如2024
        check_existing (bool): 是否检查已存在的文件，默认True
        
        返回:
        tuple: (是否成功下载数据, 下载的文件数量)
        """
        try:
            # 格式化年份
            year_str = str(year)

            # 确保本地目录存在
            location_dir = os.path.join(self.data_dir, f"locationid={location_id}/year={year_str}")
            os.makedirs(location_dir, exist_ok=True)

            # 构建S3路径
            s3_path = f"s3://openaq-data-archive/records/csv.gz/locationid={location_id}/year={year_str}/"

            logger.info(f"正在从S3下载 {year_str} 年的数据...")
            logger.info(f"S3路径: {s3_path}")
            logger.info(f"本地目录: {location_dir}")

            # 如果check_existing为True，先检查本地是否已有文件
            if check_existing:
                local_files = []
                for root, dirs, files in os.walk(location_dir):
                    local_files.extend([os.path.join(root, file) for file in files if file.endswith('.csv.gz')])

                if local_files:
                    logger.info(f"本地目录已存在 {len(local_files)} 个文件")
                    logger.info("如果需要重新下载，请使用 --force_download 选项")
                    return True, len(local_files)

            # 执行AWS CLI命令递归下载整年数据
            cmd = [
                "aws", "s3", "cp",
                "--no-sign-request",
                "--recursive",
                s3_path,
                location_dir
            ]

            logger.info(f"执行命令: {' '.join(cmd)}")

            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # 检查下载后的文件数量
                downloaded_files = []
                for root, dirs, files in os.walk(location_dir):
                    downloaded_files.extend([os.path.join(root, file) for file in files if file.endswith('.csv.gz')])

                file_count = len(downloaded_files)

                if file_count > 0:
                    logger.info(f"下载成功: 共下载了 {file_count} 个文件")
                    return True, file_count
                else:
                    logger.warning(f"警告: 下载命令执行成功，但未找到下载的文件")
                    logger.warning(f"可能S3上不存在 {year_str} 年的数据")
                    return False, 0

            except subprocess.CalledProcessError as e:
                logger.error(f"下载失败: {e.stderr}")
                return False, 0

        except Exception as e:
            logger.error(f"按年下载数据时出错: {e}")
            return False, 0

    def process_with_openaq_processor(self, df: pd.DataFrame) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[OpenAQProcessor]]:
        """
        使用OpenAQProcessor处理数据
        
        参数:
        df (pd.DataFrame): 输入数据
        
        返回:
        tuple: (处理后的数据DataFrame, 特征数据DataFrame, OpenAQProcessor实例)
        """
        if df is None or df.empty:
            logger.warning("数据为空，无法使用OpenAQProcessor处理")
            return None, None, None

        # 使用OpenAQProcessor处理数据
        processor = OpenAQProcessor()

        # 加载数据
        processor.df = df.copy()

        try:
            # 分析参数
            processor.analyze_parameters()
            # debug
            print("processor.processed_df的所有列名:", processor.df.columns.tolist())

            # 尝试填充缺失的小时
            try:
                processor.fill_missing_hours()
            except Exception as e2:
                logger.error(f"填充缺失小时时出错: {e2}")
                logger.info("跳过填充缺失小时步骤...")

            # debug
            print("processor.processed_df的所有列名:", processor.processed_df.columns.tolist())
            # 获取处理后的数据
            processed_df = processor.get_processed_data()
            logger.info(f"处理后数据形状: {processed_df.shape}")

            # 尝试计算滑动窗口特征
            features_df = None
            try:
                processor.calculate_sliding_window_features()
                features_df = processor.get_aggregated_data()
                logger.info(f"特征数据形状: {features_df.shape if features_df is not None else 'None'}")
            except Exception as e3:
                logger.error(f"计算滑动窗口特征时出错: {e3}")

            return processed_df, features_df, processor

        except Exception as e2:
            logger.error(f"手动处理数据也失败: {e2}")
            return None, None, None

    def process_aq_data_and_extract_features(
            self,
            location_id: str,
            output_dir: Optional[str] = None,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            time_freq: Optional[str] = None,
            agg_method: Union[str, Dict] = 'mean',
            download_missing: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        处理空气质量数据并提取特征
        
        参数:
        location_id (str): 位置ID
        output_dir (str, optional): 输出目录，默认为当前目录
        start_date (str, optional): 起始日期，格式为YYYYMMDD
        end_date (str, optional): 结束日期，格式为YYYYMMDD
        time_freq (str, optional): 时间重采样频率，如'1H'表示1小时，'1D'表示1天，默认为None不重采样
        agg_method (str or dict): 聚合方法，可以是字符串或字典，默认为'mean'
        download_missing (bool): 如果数据不存在，是否从AWS S3下载
        
        返回:
        tuple: (合并后的原始数据DataFrame, 处理后的DataFrame, 提取特征后的DataFrame)
        """
        # 步骤1: 确保输出目录存在
        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        # 构建输出文件名前缀
        date_range = f"{start_date}_to_{end_date}" if start_date and end_date else "all_data"
        freq_prefix = f"freq_{time_freq}_" if time_freq else ""
        file_prefix = f"loc_{location_id}_{date_range}_"

        # 步骤2: 查找并下载数据文件
        csv_files = find_data_files(self.data_dir, location_id, start_date, end_date)

        if not csv_files and download_missing and start_date and end_date:
            logger.info(f"未找到匹配的CSV文件，尝试从AWS S3下载...")
            years = get_years_from_time_range(start_date, end_date)
            file_cnt = 0
            for year in years:
                download_success, cnt = self.fetch_data_from_s3_by_year(location_id, year=year, check_existing=False)
                file_cnt += cnt
            logger.info(f"下载{file_cnt}个文件")
            csv_files = find_data_files(self.data_dir, location_id, start_date, end_date)
            if not csv_files:
                logger.warning("下载后仍未找到匹配的CSV文件，可能S3上没有该日期范围的数据")
                return None, None, None

        elif not csv_files:
            logger.warning(f"未找到匹配的CSV文件")
            return None, None, None

        logger.info(f"找到 {len(csv_files)} 个CSV文件")

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
        logger.info(f"合并后的原始数据已保存到: {raw_data_file}")

        # 步骤6: 如果指定了时间频率，进行时间重采样
        if time_freq:
            merged_df = perform_time_resampling(merged_df, time_freq, agg_method)

            # 保存重采样后的数据
            resampled_file = os.path.join(output_dir, f"{freq_prefix}resampled_{file_prefix}.csv")
            merged_df.to_csv(resampled_file, index=False)
            logger.info(f"时间重采样后的数据已保存到: {resampled_file}")

        # 步骤7: 使用OpenAQProcessor处理数据
        processed_df, features_df, _ = self.process_with_openaq_processor(merged_df)

        # 步骤8: 保存处理后的数据
        if processed_df is not None:
            hourly_file = os.path.join(output_dir, f"{freq_prefix}hourly_data_{file_prefix}.csv")
            processed_df.to_csv(hourly_file, index=False)
            logger.info(f"填充小时后的数据已保存到: {hourly_file}")

        # 步骤9: 保存特征数据
        if features_df is not None:
            features_file = os.path.join(output_dir, f"{freq_prefix}features_{file_prefix}.csv")
            features_df.to_csv(features_file, index=False)
            logger.info(f"提取的特征数据已保存到: {features_file}")

        return merged_df, processed_df, features_df

    def check_and_download_incomplete_months(
            self,
            location_id: str,
            start_year: Optional[int] = None,
            end_year: Optional[int] = None,
            start_month: Optional[int] = None,
            end_month: Optional[int] = None
    ) -> Dict:
        """
        检查并下载不完整月份的数据
        
        参数:
        location_id (str): 位置ID
        start_year (int, optional): 起始年份
        end_year (int, optional): 结束年份
        start_month (int, optional): 起始月份
        end_month (int, optional): 结束月份
        
        返回:
        Dict: 下载结果，键为(年,月)元组，值为包含成功状态和文件数的字典
        """
        # 设置默认值
        current_year = datetime.now().year
        current_month = datetime.now().month

        start_year = start_year if start_year is not None else current_year - 1
        end_year = end_year if end_year is not None else current_year

        results = {}

        # 遍历年份和月份
        for year in range(start_year, end_year + 1):
            # 确定月份范围
            if year == start_year and start_month is not None:
                months_start = start_month
            else:
                months_start = 1

            if year == end_year and end_month is not None:
                months_end = end_month
            else:
                months_end = 12 if year < current_year else current_month

            # 遍历月份
            for month in range(months_start, months_end + 1):
                month_str = f"{month:02d}"
                year_str = str(year)

                # 构建月份目录路径
                month_dir = os.path.join(self.data_dir, f"locationid={location_id}/year={year_str}/month={month_str}")

                # 检查月份目录是否存在
                if not os.path.exists(month_dir):
                    os.makedirs(month_dir, exist_ok=True)
                    logger.info(f"创建月份目录: {month_dir}")

                # 计算该月的天数
                if month == 2:
                    # 检查闰年
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                        days_in_month = 29
                    else:
                        days_in_month = 28
                elif month in [4, 6, 9, 11]:
                    days_in_month = 30
                else:
                    days_in_month = 31

                # 检查每一天的文件是否存在
                missing_days = []
                for day in range(1, days_in_month + 1):
                    day_str = f"{day:02d}"
                    file_name = f"location-{location_id}-{year_str}{month_str}{day_str}.csv.gz"
                    file_path = os.path.join(month_dir, file_name)

                    if not os.path.exists(file_path):
                        missing_days.append(day)

                # 如果有缺失的天，下载这些天的数据
                if missing_days:
                    logger.info(f"{year_str}-{month_str} 有 {len(missing_days)} 天缺失: {missing_days}")

                    # 构建日期范围
                    start_date = f"{year_str}{month_str}{missing_days[0]:02d}"
                    end_date = f"{year_str}{month_str}{missing_days[-1]:02d}"

                    # 下载缺失的天
                    success = self.download_data_from_s3(location_id, start_date, end_date)

                    # 再次检查文件数量
                    files_count = 0
                    for day in missing_days:
                        day_str = f"{day:02d}"
                        file_name = f"location-{location_id}-{year_str}{month_str}{day_str}.csv.gz"
                        file_path = os.path.join(month_dir, file_name)

                        if os.path.exists(file_path):
                            files_count += 1

                    results[(year, month)] = {
                        'success': success,
                        'files_count': files_count,
                        'missing_count': len(missing_days)
                    }
                else:
                    logger.info(f"{year_str}-{month_str} 数据完整")
                    results[(year, month)] = {
                        'success': True,
                        'files_count': days_in_month,
                        'missing_count': 0
                    }

        return results


def main():
    """主函数，处理命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='处理空气质量数据并提取特征')
    parser.add_argument('--location_id', type=str, required=True, help='位置ID')
    parser.add_argument('--data_dir', type=str,
                        help='数据根目录，包含locationid={location_id}子目录，默认为./data/opendb-aq')
    parser.add_argument('--output_dir', type=str, help='输出目录，默认为当前目录')
    parser.add_argument('--start_date', type=str, help='起始日期，格式为YYYYMMDD，如20240101')
    parser.add_argument('--end_date', type=str, help='结束日期，格式为YYYYMMDD，如20240131')
    parser.add_argument('--time_freq', type=str, help='时间重采样频率，如1H表示1小时，1D表示1天，不指定则不重采样')
    parser.add_argument('--agg_method', type=str, default='mean',
                        help='聚合方法，默认为mean，可选值：mean, median, max, min, sum')
    parser.add_argument('--download', action='store_true', help='如果数据不存在，从AWS S3下载')
    parser.add_argument('--check_incomplete', action='store_true', help='检查并下载不完整的月份数据')
    parser.add_argument('--start_year', type=int, help='检查不完整月份的起始年份')
    parser.add_argument('--end_year', type=int, help='检查不完整月份的结束年份')
    parser.add_argument('--start_month', type=int, help='检查不完整月份的起始月份')
    parser.add_argument('--end_month', type=int, help='检查不完整月份的结束月份')
    parser.add_argument('--download_year', type=int, help='下载指定年份的所有数据')
    parser.add_argument('--force_download', action='store_true', help='强制下载，不检查本地是否已存在文件')

    args = parser.parse_args()

    # 创建数据获取器
    fetcher = OpenAQDataFetcher(args.data_dir)

    # 按年下载数据
    if args.download_year is not None:
        logger.info(f"正在下载位置ID {args.location_id} 的 {args.download_year} 年数据...")
        success, file_count = fetcher.fetch_data_from_s3_by_year(
            location_id=args.location_id,
            year=args.download_year,
            check_existing=not args.force_download
        )

        status = "成功" if success else "失败"
        logger.info(f"\n下载结果: {status}, 文件数: {file_count}")
        return

    # 检查是否需要下载不完整月份
    if args.check_incomplete:
        logger.info(f"检查并下载位置ID {args.location_id} 的不完整月份数据...")
        results = fetcher.check_and_download_incomplete_months(
            location_id=args.location_id,
            start_year=args.start_year,
            end_year=args.end_year,
            start_month=args.start_month,
            end_month=args.end_month
        )

        logger.info("\n下载结果汇总:")
        for (year, month), result in sorted(results.items()):
            status = "成功" if result['success'] else "失败"
            logger.info(f"{year}-{month:02d}: {status}, 文件数: {result['files_count']}")
    else:
        # 处理聚合方法
        agg_method = args.agg_method
        if agg_method not in ['mean', 'median', 'max', 'min', 'sum']:
            logger.warning(f"不支持的聚合方法 {agg_method}，使用默认值 mean")
            agg_method = 'mean'

        fetcher.process_aq_data_and_extract_features(
            args.location_id,
            args.output_dir,
            args.start_date,
            args.end_date,
            args.time_freq,
            agg_method,
            args.download
        )


if __name__ == '__main__':
    main()
