import gzip
import pandas as pd

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
    