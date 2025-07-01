#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any, Tuple, Set
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载CSV文件，支持.csv和.csv.gz格式
    对于.gz文件，如果存在重复列名，丢弃其中值全为空的列
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    pandas.DataFrame: 加载的数据
    """
    try:
        if file_path.endswith('.gz'):
            # 首先读取列名
            with gzip.open(file_path, 'rt') as f:
                header = pd.read_csv(f, nrows=0)
                column_names = header.columns.tolist()

            # 检查是否有重复列名
            duplicate_columns = [col for col in column_names if column_names.count(col) > 1]

            if duplicate_columns:
                logger.info(f"文件 {file_path} 包含重复列名: {set(duplicate_columns)}")

                # 使用低级别的读取方式处理重复列
                with gzip.open(file_path, 'rt') as f:
                    # 使用mangle_dupe_cols=True来自动重命名重复列
                    df = pd.read_csv(f, mangle_dupe_cols=True)

                # 找出重复列（会被自动重命名为name.1, name.2等）
                renamed_duplicates = [col for col in df.columns if '.' in col and col.split('.')[0] in column_names]

                # 检查这些重复列是否全为空
                for col in renamed_duplicates:
                    if df[col].isna().all():
                        logger.info(f"删除全为空的重复列: {col}")
                        df = df.drop(columns=[col])
                    else:
                        base_col = col.split('.')[0]
                        # 如果原始列全为空而重命名列有值，则替换原始列
                        if base_col in df.columns and df[base_col].isna().all():
                            logger.info(f"原始列 {base_col} 全为空，用 {col} 替换")
                            df[base_col] = df[col]
                            df = df.drop(columns=[col])
            else:
                # 如果没有重复列，正常读取
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
        else:
            # 非gz文件正常读取
            df = pd.read_csv(file_path)

        # 删除所有全为空的列
        null_columns = [col for col in df.columns if df[col].isna().all()]
        if null_columns:
            logger.info(f"删除全为空的列: {null_columns}")
            df = df.drop(columns=null_columns)

        logger.info(f"成功加载文件: {file_path}, 形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"加载文件失败: {file_path}, 错误: {e}")
        return None


def find_data_files(data_dir: str, location_id: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> List[str]:
    """
    查找指定位置和日期范围内的数据文件
    
    参数:
    data_dir (str): 数据根目录
    location_id (str): 位置ID
    start_date (str, optional): 起始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    
    返回:
    List[str]: 符合条件的文件路径列表
    """
    # 构建查找路径
    location_path = os.path.join(data_dir, f"locationid={location_id}")

    if not os.path.exists(location_path):
        logger.warning(f"位置目录不存在: {location_path}")
        return []

    # 如果未指定日期范围，获取所有文件
    if not start_date or not end_date:
        logger.info("未指定日期范围，查找所有文件")
        return glob.glob(os.path.join(location_path, "**/*.csv.gz"), recursive=True)

    # 解析日期范围
    try:
        # 如果日期格式为"YYYY-MM-DD"或"YYYY-MM-DD "，则转换为"YYYYMMDD"
        if start_date:
            start_date = start_date.strip()
            if '-' in start_date:
                try:
                    start_date = datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y%m%d')
                except ValueError:
                    pass  # 如果不是该格式则跳过
        if end_date:
            end_date = end_date.strip()
            if '-' in end_date:
                try:
                    end_date = datetime.strptime(end_date, '%Y-%m-%d').strftime('%Y%m%d')
                except ValueError:
                    pass
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
    except ValueError as e:
        logger.error(f"日期格式错误: {e}")
        return []

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

    # 查找匹配的文件
    csv_files = []

    for (year, month), days in sorted(dates_by_year_month.items()):
        year_str = str(year)
        month_str = f"{month:02d}"

        month_path = os.path.join(location_path, f"year={year_str}", f"month={month_str}")

        if not os.path.exists(month_path):
            logger.warning(f"月份目录不存在: {month_path}")
            continue

        # 对每一天查找文件
        for day in sorted(days):
            day_str = f"{day:02d}"
            file_pattern = f"location-{location_id}-{year_str}{month_str}{day_str}.csv.gz"
            file_path = os.path.join(month_path, file_pattern)

            if os.path.exists(file_path):
                csv_files.append(file_path)
            else:
                logger.warning(f"文件不存在: {file_path}")

    logger.info(f"找到 {len(csv_files)} 个符合条件的文件")
    return csv_files


def preprocess_datetime_column(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    预处理DataFrame中的datetime列，兼容混合时区和异常值
    
    参数:
    df (pd.DataFrame): 输入数据
    datetime_col (str): 时间列名
    
    返回:
    pd.DataFrame: 处理后的数据
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法处理datetime列")
        return df

    if datetime_col not in df.columns:
        logger.warning(f"数据中没有{datetime_col}列，无法处理")
        return df

    try:
        # 1. 过滤掉明显无效的值
        df = df[df[datetime_col].notna()]
        df = df[df[datetime_col] != 'Na']

        # 2. 转为字符串，标准化格式
        df[datetime_col] = df[datetime_col].astype(str).str.replace(' ', 'T')

        # 3. 解析为datetime，强制utc，遇到异常填NaT
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', utc=True)

        # 4. 丢弃无法解析的行
        df = df.dropna(subset=[datetime_col])

        # 5. 只有在类型正确时才用.dt
        if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df['date'] = df[datetime_col].dt.date
            df['hour'] = df[datetime_col].dt.hour
        else:
            logger.error(f"{datetime_col}列不是datetimelike类型，无法提取date和hour")
            df['date'] = None
            df['hour'] = None

        logger.info(f"成功处理{datetime_col}列，提取了date和hour")
        return df

    except Exception as e:
        logger.error(f"处理{datetime_col}列时出错: {e}")
        # 兜底：返回原始df并补充空列
        if 'date' not in df.columns:
            df['date'] = None
        if 'hour' not in df.columns:
            df['hour'] = None
        return df


def sort_by_datetime(df: pd.DataFrame, datetime_col: str = 'datetime') -> pd.DataFrame:
    """
    按datetime列排序
    
    参数:
    df (pd.DataFrame): 输入数据
    datetime_col (str): 时间列名
    
    返回:
    pd.DataFrame: 排序后的数据
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法排序")
        return df

    if datetime_col not in df.columns:
        logger.warning(f"数据中没有{datetime_col}列，无法排序")
        return df

    try:
        # 确保datetime列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

        # 按datetime排序
        df = df.sort_values(by=datetime_col)

        logger.info(f"成功按{datetime_col}列排序")
        return df
    except Exception as e:
        logger.error(f"按{datetime_col}列排序时出错: {e}")
        return df


def load_and_merge_data_files(file_paths: List[str]) -> Optional[pd.DataFrame]:
    """
    加载并合并多个数据文件
    
    参数:
    file_paths (List[str]): 文件路径列表
    
    返回:
    Optional[pd.DataFrame]: 合并后的数据，如果没有有效数据则返回None
    """
    if not file_paths:
        logger.warning("没有提供文件路径")
        return None

    # 加载所有文件
    dfs = []
    for file_path in file_paths:
        df = load_csv_file(file_path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning("没有成功加载任何文件")
        return None

    # 合并数据
    try:
        merged_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"成功合并 {len(dfs)} 个文件，合并后形状: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logger.error(f"合并数据时出错: {e}")
        return None


def perform_time_resampling(df: pd.DataFrame, time_freq: str = '1H',
                            agg_method: Union[str, Dict] = 'mean',
                            time_column: str = 'datetime') -> pd.DataFrame:
    """
    按指定时间频率重采样时间序列数据
    
    参数:
    df (pd.DataFrame): 输入数据
    time_freq (str): 时间频率，如'1H'表示1小时，'1D'表示1天
    agg_method (str or dict): 聚合方法，可以是字符串或字典
    time_column (str): 时间列名
    
    返回:
    pd.DataFrame: 重采样后的数据
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法进行时间重采样")
        return df

    if time_column not in df.columns:
        logger.warning(f"数据中没有{time_column}列，无法进行时间重采样")
        return df

    try:
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

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

        logger.info(f"成功按{time_freq}频率重采样，重采样后形状: {resampled_df.shape}")
        return resampled_df
    except Exception as e:
        logger.error(f"进行时间重采样时出错: {e}")
        return df


def get_years_from_time_range(start_date: str, end_date: str) -> List[int]:
    """
    从日期范围中获取所有年份，支持两种日期格式：
    1. 'YYYY-MM-DD'
    2. 'YYYYMMDD'
    
    参数:
    start_date (str): 开始日期
    end_date (str): 结束日期
    
    返回:
    List[int]: 年份列表
    """
    try:
        # 处理两种可能的日期格式
        if '-' in start_date:
            start_year = int(start_date.split('-')[0])
        else:
            start_year = int(start_date[:4])

        if '-' in end_date:
            end_year = int(end_date.split('-')[0])
        else:
            end_year = int(end_date[:4])

        return list(range(start_year, end_year + 1))
    except (ValueError, IndexError) as e:
        logger.error(f"解析日期范围失败: {e}")
        return []


def get_unique_values(df: pd.DataFrame, column: str) -> Set:
    """
    获取DataFrame中某列的唯一值
    
    参数:
    df (pd.DataFrame): 输入数据
    column (str): 列名
    
    返回:
    Set: 唯一值集合
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法获取唯一值")
        return set()

    if column not in df.columns:
        logger.warning(f"数据中没有{column}列，无法获取唯一值")
        return set()

    try:
        unique_values = set(df[column].dropna().unique())
        logger.info(f"列{column}中有 {len(unique_values)} 个唯一值")
        return unique_values
    except Exception as e:
        logger.error(f"获取列{column}的唯一值时出错: {e}")
        return set()


def filter_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    根据条件筛选DataFrame
    
    参数:
    df (pd.DataFrame): 输入数据
    filters (Dict[str, Any]): 筛选条件，键为列名，值为筛选值
    
    返回:
    pd.DataFrame: 筛选后的数据
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法筛选")
        return df

    if not filters:
        logger.warning("未提供筛选条件")
        return df

    try:
        filtered_df = df.copy()

        for column, value in filters.items():
            if column not in df.columns:
                logger.warning(f"数据中没有{column}列，跳过此筛选条件")
                continue

            if isinstance(value, list):
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == value]

        logger.info(f"筛选前形状: {df.shape}, 筛选后形状: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logger.error(f"筛选数据时出错: {e}")
        return df


def save_dataframe(df: pd.DataFrame, file_path: str, index: bool = False) -> bool:
    """
    保存DataFrame到文件
    
    参数:
    df (pd.DataFrame): 要保存的数据
    file_path (str): 文件路径
    index (bool): 是否保存索引
    
    返回:
    bool: 是否成功保存
    """
    if df is None or df.empty:
        logger.warning("输入数据为空，无法保存")
        return False

    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        # 根据文件扩展名选择保存方式
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=index)
        elif file_path.endswith('.parquet'):
            df.to_parquet(file_path, index=index)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=index)
        else:
            # 默认保存为CSV
            df.to_csv(file_path, index=index)

        logger.info(f"成功保存数据到: {file_path}")
        return True
    except Exception as e:
        logger.error(f"保存数据时出错: {e}")
        return False
