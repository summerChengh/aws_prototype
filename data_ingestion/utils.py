import gzip
import pandas as pd
import os
import glob
from datetime import datetime

def load_csv_file(file_path):
    """
    加载CSV文件，支持.csv和.csv.gz格式
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    pandas.DataFrame: 加载的数据
    """
    try:
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None


def find_data_files(base_dir, location_id, start_date=None, end_date=None):
    """
    根据日期范围查找指定位置ID的数据文件
    
    参数:
    base_dir (str): 数据根目录
    location_id (str): 位置ID
    start_date (str, optional): 起始日期，格式为YYYYMMDD
    end_date (str, optional): 结束日期，格式为YYYYMMDD
    
    返回:
    list: 匹配条件的文件路径列表
    """
    # 解析起始和结束日期
    if start_date:
        start_dt = datetime.strptime(start_date, '%Y%m%d')
    else:
        start_dt = datetime(1900, 1, 1)  # 默认非常早的日期
        
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_dt = datetime.now()  # 默认当前日期
    
    # 构建位置ID目录路径
    location_dir = os.path.join(base_dir, f"locationid={location_id}")
    
    if not os.path.exists(location_dir):
        print(f"位置ID目录不存在: {location_dir}")
        return []
    
    # 查找所有年份目录
    year_dirs = [d for d in os.listdir(location_dir) if d.startswith('year=')]
    
    all_files = []
    
    for year_dir in sorted(year_dirs):
        year = year_dir.split('=')[1]
        year_path = os.path.join(location_dir, year_dir)
        
        # 查找所有月份目录
        month_dirs = [d for d in os.listdir(year_path) if d.startswith('month=')]
        
        for month_dir in sorted(month_dirs):
            month = month_dir.split('=')[1]
            month_path = os.path.join(year_path, month_dir)
            
            # 查找该月所有数据文件
            file_pattern = f"location-{location_id}-{year}{month}*.csv*"
            files = glob.glob(os.path.join(month_path, file_pattern))
            
            # 过滤日期范围内的文件
            for file_path in files:
                # 从文件名提取日期
                file_name = os.path.basename(file_path)
                # 文件名格式为 location-{locationid}-{year}{month}{day}.csv.gz
                date_part = file_name.split('-')[2].split('.')[0]  # 提取YYYYMMDD部分
                
                if len(date_part) >= 8:  # 确保有足够的字符
                    file_date_str = date_part[:8]  # 取前8个字符作为日期
                    try:
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')
                        
                        # 检查文件日期是否在范围内
                        if start_dt <= file_date <= end_dt:
                            all_files.append(file_path)
                    except ValueError:
                        # 如果日期解析失败，跳过该文件
                        print(f"无法解析文件日期: {file_name}")
                        continue
    
    return sorted(all_files)

def preprocess_datetime_column(df):
    """
    预处理datetime列，确保格式一致
    
    参数:
    df (pandas.DataFrame): 包含datetime列的DataFrame
    
    返回:
    pandas.DataFrame: 预处理后的DataFrame
    """
    if df is None or df.empty or 'datetime' not in df.columns:
        print("警告: 数据为空或没有datetime列")
        return df
    
    print("预处理datetime列，确保格式一致...")
    
    # 创建DataFrame的副本
    df = df.copy()
    
    # 确保datetime列是字符串类型
    df['datetime'] = df['datetime'].astype(str)
    
    # 标准化datetime格式
    # 1. 替换空格为'T'
    df['datetime'] = df['datetime'].str.replace(' ', 'T')
    
    # 2. 处理时区信息，移除时区部分以避免混合时区问题
    # 查找常见的时区模式并移除
    timezone_pattern = r'([+-]\d{2}:?\d{2}|\.\d+|Z)$'
    df['datetime'] = df['datetime'].str.replace(timezone_pattern, '', regex=True)
    
    # 3. 确保格式为 YYYY-MM-DDThh:mm:ss
    # 添加缺失的秒数
    df.loc[df['datetime'].str.count(':') == 1, 'datetime'] = df.loc[df['datetime'].str.count(':') == 1, 'datetime'] + ':00'
    
    print("datetime列预处理完成")
    return df

def sort_by_datetime(df):
    """
    按datetime列对DataFrame进行排序
    
    参数:
    df (pandas.DataFrame): 包含datetime列的DataFrame
    
    返回:
    pandas.DataFrame: 排序后的DataFrame
    """
    if df is None or df.empty or 'datetime' not in df.columns:
        print("警告: 数据为空或没有datetime列，无法排序")
        return df
    
    try:
        print("使用datetime.fromisoformat()方法排序...")
        
        # 创建DataFrame的副本
        df = df.copy()
        
        # 创建一个辅助函数，安全地将字符串转换为datetime对象
        def safe_fromisoformat(dt_str):
            try:
                # 确保字符串格式正确
                if not isinstance(dt_str, str):
                    dt_str = str(dt_str)
                
                # 处理可能的时区信息
                if '+' in dt_str:
                    dt_str = dt_str.split('+')[0]
                if '-' in dt_str and dt_str.count('-') > 2:  # 有时区的减号
                    dt_str = dt_str.rsplit('-', 1)[0]
                
                # 尝试使用fromisoformat解析
                return datetime.fromisoformat(dt_str)
            except ValueError:
                # 如果解析失败，尝试更宽松的解析方法
                try:
                    return pd.to_datetime(dt_str)
                except:
                    # 如果仍然失败，返回一个很早的日期作为默认值
                    print(f"无法解析日期时间: {dt_str}")
                    return datetime(1900, 1, 1)
        
        # 创建一个排序键列
        df['sort_key'] = df['datetime'].apply(safe_fromisoformat)
        
        # 按排序键排序
        df = df.sort_values('sort_key')
        
        # 删除辅助列
        df = df.drop(columns=['sort_key'])
        
        print("成功使用datetime.fromisoformat()方法排序")
        return df
        
    except Exception as e:
        print(f"使用datetime.fromisoformat()排序时出错: {e}")
        print("回退到默认排序方法...")
        return df.sort_values('datetime')

def load_and_merge_data_files(csv_files):
    """
    加载并合并多个CSV文件
    
    参数:
    csv_files (list): CSV文件路径列表
    
    返回:
    pandas.DataFrame: 合并后的DataFrame
    """
    if not csv_files:
        print("没有提供CSV文件")
        return None
    
    all_data = []
    for file_path in csv_files:
        print(f"处理文件: {os.path.basename(file_path)}")
        df = load_csv_file(file_path)
        if df is not None and not df.empty:
            # 确保datetime列保持为字符串类型
            if 'datetime' in df.columns:
                df['datetime'] = df['datetime'].astype(str)
            all_data.append(df)
    
    if not all_data:
        print("没有有效数据可处理")
        return None
    
    # 合并所有数据
    merged_df = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据形状: {merged_df.shape}")
    return merged_df

def perform_time_resampling(df, time_freq, agg_method='mean'):
    """
    按指定时间频率重采样数据
    
    参数:
    df (pandas.DataFrame): 输入数据
    time_freq (str): 时间频率，如'1H'表示1小时，'1D'表示1天
    agg_method (str or dict): 聚合方法，可以是字符串或字典
    
    返回:
    pandas.DataFrame: 重采样后的数据
    """
    if df is None or df.empty:
        print("警告: 数据为空，无法进行时间重采样")
        return df
    
    if 'datetime' not in df.columns:
        print("警告: 数据中没有datetime列，无法进行时间重采样")
        return df
    
    print(f"按 {time_freq} 时间频率重采样数据...")
    
    try:
        # 创建DataFrame的副本
        df = df.copy()
        
        # 对于重采样，需要将datetime转换为datetime类型，设置utc=True避免混合时区警告
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        
        # 进行时间重采样
        resampled_df = resample_time_series(df, time_freq=time_freq, agg_method=agg_method)
        
        # 重采样后，将datetime列转回字符串以便OpenAQProcessor处理
        if 'datetime' in resampled_df.columns:
            resampled_df['datetime'] = resampled_df['datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        print(f"时间重采样后数据形状: {resampled_df.shape}")
        return resampled_df
        
    except Exception as e:
        print(f"时间重采样失败: {e}")
        print("返回原始数据...")
        return df


def get_years_from_time_range(start_date, end_date):
    """
    从起始日期到结束日期解析出所有年份
    
    参数:
    start_date (str): 起始日期，格式为 'YYYY-MM-DD'
    end_date (str): 结束日期，格式为 'YYYY-MM-DD'
    
    返回:
    list: 包含在日期范围内的所有年份的列表
    """
    import datetime
    
    # 解析日期字符串为datetime对象
    try:
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"日期格式错误: {e}")
        return []
    
    # 获取起始年份和结束年份
    start_year = start.year
    end_year = end.year
    
    # 返回年份列表
    return list(range(start_year, end_year + 1))
    