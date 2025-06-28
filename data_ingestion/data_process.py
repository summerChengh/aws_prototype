# 导入所需的库和函数
import os
import pandas as pd
import random
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fetch_noaa_data import getNoaaDataFrame, get_noaa_location_ids, get_stations_info_from_df, get_openaq_stations
from utils import get_years_from_time_range
from fetch_openaq_data import process_aq_data_and_extract_features
from cal_aqi import AQICalculator

# 指定时间范围，城市采样比例
start_date = "2024-01-01"  # 格式为 'YYYY-MM-DD'，用于NOAA数据
end_date = "2024-01-31"    # 格式为 'YYYY-MM-DD'，用于NOAA数据
sample_rate = 0.001

# 为OpenAQ数据转换日期格式
start_date_openaq = start_date.replace("-", "")  # 转换为'YYYYMMDD'格式
end_date_openaq = end_date.replace("-", "")      # 转换为'YYYYMMDD'格式

# 确保输出目录存在
output_dir = f"./data/processed/{start_date_openaq}_{end_date_openaq}"
os.makedirs(output_dir, exist_ok=True)

# 解析nooa的所有locations
years = get_years_from_time_range(start_date, end_date)
#station_ids = get_noaa_location_ids(years)

# 采样nooa locations
# 确保采样数量至少为1
#sample_size = max(1, int(len(station_ids) * sample_rate))
#sampled_stations = random.sample(station_ids, sample_size)
#print(f"采样了 {sample_size} 个站点，从总共 {len(station_ids)} 个站点中")

# 获取指定时间范围所有的nooa数据
#nooa_df = getNoaaDataFrame(sampled_stations, start_date, end_date)
nooa_file = f"{output_dir}/nooa.csv"
#nooa_df.to_csv(nooa_file, index=False)
nooa_df = pd.read_csv(nooa_file)
print(f"已保存NOAA数据到: {nooa_file}")

# 从DataFrame中获取stations经纬度信息
stations_file = f"{output_dir}/nooa_stations_file.csv"
#get_stations_info_from_df(nooa_df, stations_file)
print(f"已保存站点信息到: {stations_file}")

# 采样后的nooa locations获取最近的openaq location ids
station_location_file = f"{output_dir}/noaa-openaq-mapping_test.csv"
#get_openaq_stations(stations_file, station_location_file, api_key="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00")

# 获取openaq locations ids指定时间范围的数据，并计算污染物指标
# 加载station_location_file，获取所有不重复的OPENAQ_ID
mapping_df = pd.read_csv(station_location_file)
unique_openaq_ids = mapping_df['OPENAQ_ID'].dropna().unique()
print(f"共找到 {len(unique_openaq_ids)} 个不重复的OPENAQ_ID")

# 合并所有id的processed_df到一张表
all_processed_dfs = []

for id in unique_openaq_ids:
    merged_aq_df, _, processed_df = process_aq_data_and_extract_features(
        id, 
        output_dir=f"{output_dir}/{id}", 
        start_date=start_date_openaq,  # 使用转换后的日期格式
        end_date=end_date_openaq,       # 使用转换后的日期格式
        download_missing=True
    )
    print("merged_aq_df columns:", merged_aq_df.columns.tolist())
    print("processed_df columns:", processed_df.columns.tolist())
    if processed_df is not None and not processed_df.empty:
        # 增加id列以便区分
        processed_df['OPENAQ_ID'] = id
        all_processed_dfs.append(processed_df)
if all_processed_dfs:
    merged_df = pd.concat(all_processed_dfs, ignore_index=True)
    merged_file = f"{output_dir}/all_openaq_processed.csv"
    merged_df.to_csv(merged_file, index=False)
    print(f"所有id的processed_df已合并并保存到: {merged_file}")
else:
    print("没有可合并的processed_df，未生成合并文件")

# 计算AQI   
# 创建AQI计算器实例
calculator = AQICalculator()

# 读取合并后的数据
merged_file = f"{output_dir}/all_openaq_processed.csv"
merged_df = pd.read_csv(merged_file)

# 创建用于存储单个污染物AQI的列
pollutants = {
    'pm2.5_24h': 'PM2.5_24h',
    'pm10_24h': 'PM10_24h',
    'o3_8h': 'O3_8h',
    'o3_1h': 'O3_1h',
    'co_8h': 'CO_8h',
    'so2_1h': 'SO2_1h',
    'so2_24h': 'SO2_24h',
    'no2_1h': 'NO2_1h'
}

# 计算综合AQI
def calculate_row_aqi(row):
    # 构建浓度字典
    concentrations = {}
    for col, pollutant_type in pollutants.items():
        if col in row.index and pd.notnull(row[col]):
            concentrations[pollutant_type] = row[col]
    
    # 如果没有有效的污染物数据，返回None
    if not concentrations:
        return None, None, {}
    
    # 计算综合AQI和单个污染物AQI
    overall_aqi, main_pollutant, single_aqis = calculator.calculate_overall_aqi(concentrations)
    return overall_aqi, main_pollutant, single_aqis

# 应用函数到每一行
aqi_results = merged_df.apply(calculate_row_aqi, axis=1)
print(aqi_results)
# 提取综合AQI和主要污染物
merged_df['AQI'] = [result[0] if result else None for result in aqi_results]
merged_df['main_pollutant'] = [result[1] if result else None for result in aqi_results]

# 添加AQI类别和描述
merged_df['AQI_category'] = merged_df['AQI'].apply(
    lambda x: calculator.get_aqi_category(x)[0] if pd.notnull(x) else None
)

# 提取单个污染物的AQI值
# 从single_aqis字典中提取各污染物的AQI值
for col in ["O3", "SO2", "PM2.5_24h", "PM10_24h", "CO_8h", "NO2_1h"]:
    # 创建新列来存储该污染物的AQI
    aqi_col = f"{col}_aqi"

    merged_df[aqi_col] = [result[2][col][1] if col in result[2] else None for result in aqi_results]
    
    print(f"已从calculate_overall_aqi结果中提取{col}的AQI并保存到{aqi_col}列")

# 保存带有AQI的数据
aqi_file = f"{output_dir}/all_openaq_with_aqi.csv"
merged_df.to_csv(aqi_file, index=False)
print(f"已计算AQI并保存到: {aqi_file}")

# 将nooa_df和merged_df进行左连接，使用mapping_df中的STATION和OPENAQ_ID字段的对应关系
print(merged_df.columns)
feats = ["location_id", "date", "location", "lat", "lon", "O3_aqi", "SO2_aqi", "PM2.5_24h_aqi", "PM10_24h_aqi",
 "CO_8h_aqi", "NO2_1h_aqi", "AQI", 'main_pollutant', "OPENAQ_ID"]
merged_df = merged_df[feats]
# 1. 先将mapping_df中的STATION和OPENAQ_ID作为桥接
# 2. nooa_df中应有STATION字段，merged_df中有OPENAQ_ID字段
# 确保字段名一致
nooa_df_merged = nooa_df.copy()
if 'STATION' not in nooa_df_merged.columns:
    # 有些nooa数据可能用'station'小写
    if 'station' in nooa_df_merged.columns:
        nooa_df_merged.rename(columns={'station': 'STATION'}, inplace=True)

# 只保留mapping_df中有用的两列
mapping_bridge = mapping_df[['STATION', 'OPENAQ_ID']].dropna()

# 先将nooa_df和mapping_df左连接，获得OPENAQ_ID
nooa_with_openaq = pd.merge(nooa_df_merged, mapping_bridge, on='STATION', how='left')

# 再与merged_df左连接，on OPENAQ_ID 和时间字段（如有需要可加时间对齐，这里只按OPENAQ_ID连接）
# 如果需要按时间对齐，可以加上时间字段，比如 'datetime' 或 'date'
# 这里仅按OPENAQ_ID左连接，保留nooa数据的所有行
final_merged = pd.merge(nooa_with_openaq, merged_df, on='OPENAQ_ID', how='left', suffixes=('_nooa', '_openaq'))
print(final_merged.columns)
# 保存最终合并结果
final_merged_file = f"{output_dir}/nooa_openaq_merged.csv"
final_merged.to_csv(final_merged_file, index=False)
print(f"NOAA与OpenAQ数据已左连接并保存到: {final_merged_file}")

print("数据处理完成")
