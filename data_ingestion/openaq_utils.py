import requests
import pandas as pd
import numpy as np
import argparse
import os
from typing import List, Dict, Optional, Union, Any
from utils import get_years_from_time_range
from datetime import datetime, timedelta


def fetch_openaq_location_ids(lat, lng, api_key=None, radius_m=10000, limit=10):
    """
    使用OpenAQ API v3获取指定位置周围的监测站点ID
    OpenAQ API v3需要API Key进行认证
    """
    # 使用OpenAQ API v3
    aq_reqUrlBase = "https://api.openaq.org/v3"
    aq_reqParams = {
        'limit': limit,
        'page': 1,
        'offset': 0,
        'sort': 'desc',
        'parameter': None,
        'coordinates': f'{lat},{lng}',
        'radius': radius_m,
        'isMobile': 'false',
        'sensorType': 'reference grade'
    }
    # 设置请求头，包含API Key
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    # 正确调用locations端点
    try:
        resp = requests.get(f"{aq_reqUrlBase}/locations", params=aq_reqParams, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # 调试输出
        print(f"API响应状态码: {resp.status_code}")
        print(f"API请求URL: {resp.url}")
        
        # 提取location IDs (注意v3 API可能有不同的响应结构)
        ids = []
        if 'results' in data and len(data['results']) > 0:
            ids = [item['id'] for item in data['results']]
        return ids
    except requests.exceptions.HTTPError as e:
        print(f"API请求错误: {e}")
        if resp.status_code == 401:
            print("认证失败: 请提供有效的API Key")
        elif resp.status_code == 403:
            print("权限不足: 请检查API Key权限")
        return []


def get_location_by_id(location_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    根据location_id获取OpenAQ监测站点的详细信息
    
    参数:
    location_id (str): OpenAQ监测站点ID
    api_key (str, optional): OpenAQ API密钥
    
    返回:
    Dict[str, Any]: 包含监测站点详细信息的字典，如果获取失败则返回空字典
    """
    # OpenAQ API v3基础URL
    aq_reqUrlBase = "https://api.openaq.org/v3"
    
    # 设置请求头，包含API Key
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    try:
        # 调用locations/{id}端点获取特定站点信息
        resp = requests.get(f"{aq_reqUrlBase}/locations/{location_id}", headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # 调试输出
        print(f"API响应状态码: {resp.status_code}")
        
        # 返回结果数据
        if 'results' in data and len(data['results']) > 0:
            return data['results'][0]
        else:
            print(f"未找到ID为{location_id}的监测站点")
            return {}
    except requests.exceptions.HTTPError as e:
        print(f"API请求错误: {e}")
        if resp.status_code == 401:
            print("认证失败: 请提供有效的API Key")
        elif resp.status_code == 403:
            print("权限不足: 请检查API Key权限")
        elif resp.status_code == 404:
            print(f"监测站点不存在: ID {location_id} 未找到")
        return {}
    except Exception as e:
        print(f"获取监测站点信息时出错: {str(e)}")
        return {}

def get_sensors_by_location_id(location_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定位置ID的传感器列表
    
    参数:
    location_id (str): OpenAQ监测站点ID
    api_key (str, optional): OpenAQ API密钥
    
    返回:
    Dict[str, Any]: 包含传感器信息的字典，如果获取失败则返回空字典
    """
    # OpenAQ API v3基础URL
    aq_reqUrlBase = "https://api.openaq.org/v3"
    
    # 设置请求头，包含API Key
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    try:
        # 调用locations/{location_id}/sensors端点获取传感器信息
        endpoint = f"{aq_reqUrlBase}/locations/{location_id}/sensors"
        print(f"请求URL: {endpoint}")
        
        resp = requests.get(endpoint, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # 调试输出
        print(f"API响应状态码: {resp.status_code}")
        
        # 返回结果数据
        if 'results' in data and len(data['results']) > 0:
            print(f"找到{len(data['results'])}个传感器")
            return data
        else:
            print(f"未找到ID为{location_id}的监测站点的传感器")
            return {"results": []}
    except requests.exceptions.HTTPError as e:
        print(f"API请求错误: {e}")
        if resp.status_code == 401:
            print("认证失败: 请提供有效的API Key")
        elif resp.status_code == 403:
            print("权限不足: 请检查API Key权限")
        elif resp.status_code == 404:
            print(f"监测站点不存在: ID {location_id} 未找到")
        return {"results": []}
    except Exception as e:
        print(f"获取传感器信息时出错: {str(e)}")
        return {"results": []}

def get_measurements_by_sensor_id(sensor_id: int, datetime_from: Optional[str] = None, 
                             datetime_to: Optional[str] = None, limit: int = 100, 
                             page: int = 1, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取传感器按小时到日期的聚合测量数据
    
    参数:
    sensor_id (int): 传感器ID
    datetime_from (str, optional): 开始日期时间，格式为'YYYY-MM-DD'
    datetime_to (str, optional): 结束日期时间，格式为'YYYY-MM-DD'
    limit (int, optional): 返回结果数量限制，默认100
    page (int, optional): 分页页码，默认1
    api_key (str, optional): OpenAQ API密钥
    
    返回:
    Dict[str, Any]: 包含传感器测量数据的字典，如果获取失败则返回空字典
    """
    # OpenAQ API v3基础URL
    aq_reqUrlBase = "https://api.openaq.org/v3"
    
    # 确保sensor_id是整数
    try:
        sensor_id = int(sensor_id)
    except (ValueError, TypeError):
        print(f"错误: sensor_id必须是整数，收到的值为: {sensor_id}")
        return {"results": []}
    
    # 设置请求头，包含API Key
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    # 构建查询参数
    params = {
        'limit': limit,
        'page': page
    }
    
    # 添加可选的日期时间参数
    if datetime_from:
        params['datetime_from'] = datetime_from
    if datetime_to:
        params['datetime_to'] = datetime_to
    
    try:
        # 调用sensors/{sensor_id}/hours/daily端点获取数据
        endpoint = f"{aq_reqUrlBase}/sensors/{sensor_id}/hours/daily"
        print(f"请求URL: {endpoint}")
        print(f"参数: {params}")
        
        resp = requests.get(endpoint, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # 调试输出
        print(f"API响应状态码: {resp.status_code}")
        print(f"sensor resp result: {data}")
        # 返回结果数据
        if 'results' in data and len(data['results']) > 0:
            print(f"找到{len(data['results'])}条测量记录")
            return data
        else:
            print(f"未找到ID为{sensor_id}的传感器测量数据")
            return {"results": []}
    except requests.exceptions.HTTPError as e:
        print(f"API请求错误: {e}")
        if hasattr(resp, 'status_code'):
            if resp.status_code == 401:
                print("认证失败: 请提供有效的API Key")
            elif resp.status_code == 403:
                print("权限不足: 请检查API Key权限")
            elif resp.status_code == 404:
                print(f"传感器不存在: ID {sensor_id} 未找到")
        return {"results": []}
    except Exception as e:
        print(f"获取传感器测量数据时出错: {str(e)}")
        return {"results": []}

def get_latest_measurements_by_location(location_id: int, limit: int = 100, page: int = 1, 
                                   datetime_min: Optional[str] = None, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定位置的最新测量数据
    
    参数:
    location_id (int): 位置ID
    limit (int, optional): 返回结果数量限制，默认100
    page (int, optional): 分页页码，默认1
    datetime_min (str, optional): 最小日期时间，格式为'YYYY-MM-DDThh:mm:ss'
    api_key (str, optional): OpenAQ API密钥
    
    返回:
    Dict[str, Any]: 包含位置最新测量数据的字典，如果获取失败则返回空字典
    """
    # OpenAQ API v3基础URL
    aq_reqUrlBase = "https://api.openaq.org/v3"
    
    # 确保location_id是整数
    try:
        location_id = int(location_id)
    except (ValueError, TypeError):
        print(f"错误: location_id必须是整数，收到的值为: {location_id}")
        return {"results": []}
    
    # 设置请求头，包含API Key
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key
    
    # 构建查询参数
    params = {
        'limit': limit,
        'page': page
    }
    
    # 添加可选的最小日期时间参数
    if datetime_min:
        params['datetime_min'] = datetime_min
    
    try:
        # 调用locations/{location_id}/latest端点获取数据
        endpoint = f"{aq_reqUrlBase}/locations/{location_id}/latest"
        print(f"请求URL: {endpoint}")
        print(f"参数: {params}")
        
        resp = requests.get(endpoint, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        
        # 调试输出
        print(f"API响应状态码: {resp.status_code}")
        
        # 返回结果数据
        if 'results' in data and len(data['results']) > 0:
            print(f"找到{len(data['results'])}条最新测量记录")
            return data
        else:
            print(f"未找到ID为{location_id}的位置的最新测量数据")
            return {"results": []}
    except requests.exceptions.HTTPError as e:
        print(f"API请求错误: {e}")
        if hasattr(resp, 'status_code'):
            if resp.status_code == 401:
                print("认证失败: 请提供有效的API Key")
            elif resp.status_code == 403:
                print("权限不足: 请检查API Key权限")
            elif resp.status_code == 404:
                print(f"位置不存在: ID {location_id} 未找到")
        return {"results": []}
    except Exception as e:
        print(f"获取位置最新测量数据时出错: {str(e)}")
        return {"results": []}


if __name__=='__main__':
   # 创建命令行参数解析器
   parser = argparse.ArgumentParser(description='测试OpenAQ API工具函数')
   parser.add_argument('--test', type=str, choices=['locations', 'sensors', 'measurements', 'latest', 'all'], 
                      default='measurements', help='要测试的功能')
   parser.add_argument('--api-key', type=str, default="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00", 
                      help='OpenAQ API密钥')
   parser.add_argument('--location-id', type=str, default="6868", help='位置ID')
   parser.add_argument('--sensor-id', type=int, default=1662909, help='传感器ID')
   parser.add_argument('--lat', type=float, default=35.43, help='纬度')
   parser.add_argument('--lng', type=float, default=-119.01, help='经度')
   parser.add_argument('--days', type=int, default=7, help='获取最近几天的数据')
   
   args = parser.parse_args()
   
   # 计算日期范围
   end_date = datetime.now().strftime("%Y-%m-%d")
   start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
   
   print(f"测试时间范围: {start_date} 到 {end_date}")
   
   # 测试获取监测站点ID
   if args.test in ['locations', 'all']:
       print("\n===== 测试获取监测站点ID =====")
       ids = fetch_openaq_location_ids(args.lat, args.lng, api_key=args.api_key)
       print(f"找到的监测站点IDs: {ids}")
       
       if ids:
           for id in ids:
               location_info = get_location_by_id(id, api_key=args.api_key)
               print("\n监测站点详细信息:")
               print(f"id: {id}")
               print(f"名称: {location_info.get('name', 'N/A')}")
               print(f"城市: {location_info.get('city', 'N/A')}")
               print(f"国家: {location_info.get('country', 'N/A')}")
               print(f"坐标: {location_info.get('coordinates', {}).get('latitude', 'N/A')}, {location_info.get('coordinates', {}).get('longitude', 'N/A')}")
               print(f"参数: {', '.join([p.get('parameter', 'N/A') for p in location_info.get('parameters', [])])}")
   
   # 测试获取传感器信息
   if args.test in ['sensors', 'all']:
       print("\n===== 测试获取传感器信息 =====")
       location_id = args.location_id
       print(f"获取位置ID {location_id} 的传感器信息:")
       sensors_data = get_sensors_by_location_id(location_id, api_key=args.api_key)
       
       # 存储找到的传感器ID
       sensor_ids = []
       
       if 'results' in sensors_data and sensors_data['results']:
           for i, sensor in enumerate(sensors_data['results']):
               print(f"\n传感器 #{i+1}:")
               sensor_id = sensor.get('id', 'N/A')
               print(f"ID: {sensor_id}")
               print(f"参数: {sensor.get('parameter', 'N/A')}")
               print(f"类型: {sensor.get('sensor_type', 'N/A')}")
               if 'last_value' in sensor:
                   print(f"最新值: {sensor['last_value']} {sensor.get('unit', '')}")
                   print(f"最后更新时间: {sensor.get('last_updated', 'N/A')}")
               
               # 保存传感器ID
               if sensor_id != 'N/A':
                   sensor_ids.append(sensor_id)
       else:
           print("未找到传感器数据")
   
   # 测试获取传感器测量数据
   if args.test in ['measurements', 'all']:
       print("\n===== 测试获取传感器测量数据 =====")
       
       # 使用命令行参数中的传感器ID
       sensor_id = args.sensor_id
       
       print(f"\n测试 1: 获取传感器 {sensor_id} 的指定日期范围数据")
       print(f"日期范围: {start_date} 到 {end_date}")
       
       measurements = get_measurements_by_sensor_id(
           sensor_id, 
           datetime_from=start_date,
           datetime_to=end_date,
           limit=100,
           api_key=args.api_key
       )
       
       if 'results' in measurements and measurements['results']:
           print(f"找到 {len(measurements['results'])} 条测量记录")
           for i, record in enumerate(measurements['results'][:5]):  # 显示前5条记录
               print(f"\n记录 #{i+1}:")
               print(f"日期: {record.get('day', 'N/A')}")
               print(f"平均值: {record.get('average', 'N/A')}")
               print(f"最小值: {record.get('minimum', 'N/A')}")
               print(f"最大值: {record.get('maximum', 'N/A')}")
               print(f"单位: {record.get('unit', 'N/A')}")
       else:
           print("未找到测量数据")
       
       print(f"\n测试 2: 只获取最近3天的数据")
       recent_start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
       print(f"日期范围: {recent_start} 到 {end_date}")
       
       recent_measurements = get_measurements_by_sensor_id(
           sensor_id, 
           datetime_from=recent_start,
           datetime_to=end_date,
           limit=100,
           api_key=args.api_key
       )
       
       if 'results' in recent_measurements and recent_measurements['results']:
           print(f"找到 {len(recent_measurements['results'])} 条最近测量记录")
           for i, record in enumerate(recent_measurements['results']):
               print(f"\n记录 #{i+1}:")
               print(f"日期: {record.get('day', 'N/A')}")
               print(f"平均值: {record.get('average', 'N/A')}")
               print(f"最小值: {record.get('minimum', 'N/A')}")
               print(f"最大值: {record.get('maximum', 'N/A')}")
               print(f"单位: {record.get('unit', 'N/A')}")
       else:
           print("未找到最近测量数据")
       
       print(f"\n测试 3: 测试分页功能")
       print(f"获取第一页 (limit=5):")
       
       page1_measurements = get_measurements_by_sensor_id(
           sensor_id, 
           datetime_from=start_date,
           datetime_to=end_date,
           limit=5,
           page=1,
           api_key=args.api_key
       )
       
       if 'results' in page1_measurements and page1_measurements['results']:
           print(f"第一页找到 {len(page1_measurements['results'])} 条记录")
           for i, record in enumerate(page1_measurements['results']):
               print(f"记录 #{i+1}: {record.get('day', 'N/A')} - 平均值: {record.get('average', 'N/A')}")
       else:
           print("第一页未找到数据")
       
       print(f"\n获取第二页 (limit=5):")
       
       page2_measurements = get_measurements_by_sensor_id(
           sensor_id, 
           datetime_from=start_date,
           datetime_to=end_date,
           limit=5,
           page=2,
           api_key=args.api_key
       )
       
       if 'results' in page2_measurements and page2_measurements['results']:
           print(f"第二页找到 {len(page2_measurements['results'])} 条记录")
           for i, record in enumerate(page2_measurements['results']):
               print(f"记录 #{i+1}: {record.get('day', 'N/A')} - 平均值: {record.get('average', 'N/A')}")
       else:
           print("第二页未找到数据")
   
   # 测试获取位置的最新测量数据
   if args.test in ['latest', 'all']:
       print("\n===== 测试获取位置的最新测量数据 =====")
       location_id = args.location_id
       
       print(f"获取位置ID {location_id} 的最新测量数据:")
       latest_data = get_latest_measurements_by_location(
           location_id,
           limit=10,
           api_key=args.api_key
       )
       
       if 'results' in latest_data and latest_data['results']:
           print(f"找到 {len(latest_data['results'])} 条最新测量记录")
           for i, measurement in enumerate(latest_data['results']):
               print(f"\n测量 #{i+1}:")
               print(f"sensorsId: {measurement.get('sensorsId', 'N/A')}")
               print(f"参数: {measurement.get('parameter', 'N/A')}")
               print(f"值: {measurement.get('value', 'N/A')} {measurement.get('unit', '')}")
               print(f"日期时间: {measurement.get('datetime', {}).get('utc', 'N/A')}")
       else:
           print("未找到最新测量数据")
   
   print("\n测试完成!")