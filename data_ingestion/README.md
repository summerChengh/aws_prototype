# OpenAQ 数据处理工具

这个工具用于处理来自 OpenAQ API 的空气质量数据，包括获取、处理和填充缺失的小时数据。

## 功能特点

- 从CSV文件加载OpenAQ数据
- 解析日期时间，处理不同的时区格式
- 提取小时信息
- 对每个监测站点、参数和日期的组合，填充缺失的小时数据
- 支持链式调用的面向对象API
- 保持与旧版函数的兼容性

## 安装依赖

```bash
pip install pandas numpy
```

## 使用方法

### 方法1：使用链式调用（推荐）

```python
from openaq import OpenAQProcessor

# 创建处理器
processor = OpenAQProcessor(fill_value=0.0)

# 链式调用处理数据
processed_df = processor.load_data("sensor_data.csv") \
                       .parse_datetime() \
                       .analyze_parameters() \
                       .fill_missing_hours() \
                       .get_processed_data()

# 保存处理后的数据
processor.save_data("processed_data.csv")
```

### 方法2：使用便捷方法

```python
from openaq import OpenAQProcessor

# 创建处理器并一步处理数据
processor = OpenAQProcessor(fill_value=0.0)
processed_df = processor.process_file("sensor_data.csv", "processed_data.csv")
```

### 方法3：使用兼容性函数（与旧版代码兼容）

```python
from openaq import load_and_process_openaq_data

# 一步处理数据
processed_df = load_and_process_openaq_data("sensor_data.csv", fill_value=0.0)
```

### 命令行使用

```bash
# 基本使用
python openaq.py --input sensor_data.csv --output processed_data.csv

# 使用0填充缺失值
python openaq.py --input sensor_data.csv --output processed_data.csv --fill_zero

# 使用自定义值填充缺失值
python openaq.py --input sensor_data.csv --output processed_data.csv --fill_value -1
```

## 示例

查看 `process_example.py` 文件获取完整示例：

```bash
python process_example.py
```

## 数据格式

该工具期望的CSV文件格式应包含以下列：

- `datetime`: 日期时间列，如 "2024-01-01T01:00:00-0800"
- `parameter`: 空气质量参数，如 "o3", "pm25" 等
- `value`: 测量值
- `location` 或 `location_id`: 监测站点标识符

其他列将被保留并在填充缺失小时时复制。

## 自定义

`OpenAQProcessor` 类提供了多个方法，可以在处理流程的不同阶段进行自定义：

- `load_data`: 加载CSV数据
- `parse_datetime`: 解析日期时间列
- `analyze_parameters`: 分析参数
- `fill_missing_hours`: 填充缺失的小时数据
- `get_processed_data`: 获取处理后的数据
- `save_data`: 保存处理后的数据
- `process_file`: 便捷方法，执行完整处理流程 