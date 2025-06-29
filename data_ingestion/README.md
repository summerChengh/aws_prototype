# 空气质量数据处理工具

这个工具集用于处理来自 NOAA 和 OpenAQ 的气象和空气质量数据，包括数据获取、处理、特征提取和AQI计算。

## 功能特点

- 获取 NOAA 气象数据（温度、湿度、气压等）
- 获取 OpenAQ 空气质量数据（PM2.5、PM10、O3、NO2、SO2、CO等）
- 基于地理位置将 NOAA 站点与最近的 OpenAQ 站点匹配
- 处理和填充缺失的小时数据
- 计算各污染物的AQI指数和综合AQI
- 提取空气质量特征
- 将气象数据与空气质量数据合并

## 安装依赖

```bash
pip install pandas numpy requests
```

## 数据处理流程

### 1. 设置时间范围和采样参数

```python
start_date = "2016-12-01"  # 格式为 'YYYY-MM-DD'
end_date = "2016-12-30"    # 格式为 'YYYY-MM-DD'
```

### 2. 获取 NOAA 气象数据

```python
# 获取指定站点和时间范围的NOAA数据
nooa_df = getNoaaDataFrame(sampled_stations, start_date, end_date)
```

### 3. 提取站点信息并映射到 OpenAQ 站点

```python
# 从DataFrame中获取stations经纬度信息
get_stations_info_from_df(nooa_df, stations_file)

# 将NOAA站点与最近的OpenAQ站点匹配
get_openaq_stations(stations_file, station_location_file, api_key="your_api_key")
```

### 4. 获取并处理 OpenAQ 空气质量数据

```python
# 对每个OpenAQ站点处理数据
for id in unique_openaq_ids:
    merged_aq_df, _, processed_df = process_aq_data_and_extract_features(
        id, 
        output_dir=f"{output_dir}/{id}", 
        start_date=start_date_openaq,
        end_date=end_date_openaq,
        download_missing=True
    )
```

### 5. 计算AQI指数

```python
# 创建AQI计算器实例
calculator = AQICalculator()

# 计算综合AQI和各污染物AQI
overall_aqi, main_pollutant, single_aqis = calculator.calculate_overall_aqi(concentrations)
```

### 6. 合并气象和空气质量数据

```python
# 使用NOAA站点和OpenAQ站点的映射关系合并数据
nooa_with_openaq = pd.merge(nooa_df_merged, mapping_bridge, on='STATION', how='left')
final_merged = pd.merge(nooa_with_openaq, merged_df, on='OPENAQ_ID', how='left')
```

## 主要模块说明

### fetch_noaa_data.py

用于获取NOAA气象数据，包括温度、湿度、气压等气象参数。

```python
# 获取NOAA站点ID列表
station_ids = get_noaa_location_ids(years)

# 获取指定站点和时间范围的数据
nooa_df = getNoaaDataFrame(station_ids, start_date, end_date)
```

### openaq_utils.py

用于与OpenAQ API交互，获取空气质量数据。

```python
# 将NOAA站点与最近的OpenAQ站点匹配
get_openaq_stations(stations_file, output_file, api_key)
```

### fetch_openaq_data.py

处理OpenAQ数据，包括下载、合并和特征提取。

```python
# 处理特定站点的数据并提取特征
merged_aq_df, _, processed_df = process_aq_data_and_extract_features(
    location_id, output_dir, start_date, end_date, download_missing
)
```

### cal_aqi.py

计算空气质量指数(AQI)和相关类别。

```python
# 创建AQI计算器
calculator = AQICalculator()

# 计算AQI
aqi, pollutant, single_aqis = calculator.calculate_overall_aqi(concentrations)

# 获取AQI类别
category, description = calculator.get_aqi_category(aqi)
```

### openaq_feats.py

从原始数据中提取空气质量特征。

## 输出文件

处理过程会生成以下文件：

- `nooa.csv`: NOAA气象数据
- `nooa_stations_file.csv`: NOAA站点信息
- `noaa-openaq-mapping.csv`: NOAA站点与OpenAQ站点的映射关系
- `all_openaq_processed.csv`: 处理后的所有OpenAQ数据
- `all_openaq_with_aqi.csv`: 带有AQI计算结果的OpenAQ数据
- `nooa_openaq_merged.csv`: 最终合并的气象和空气质量数据

## 使用示例

完整的数据处理流程可以通过运行 `data_process.py` 实现：

```bash
python data_process.py
```

这将执行从数据获取到最终合并的完整流程，并在指定的输出目录生成所有中间文件和最终结果。 