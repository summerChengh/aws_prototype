import requests
import pandas as pd
import numpy as np
import argparse
import os
from typing import List, Dict, Optional, Union, Any


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


if __name__=='__main__':
   # 测试获取监测站点ID
   ids = fetch_openaq_location_ids(35.43, -119.01, api_key="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00")
   print(f"找到的监测站点IDs: {ids}")
   
   # 如果找到了监测站点，测试获取第一个监测站点的详细信息
   if ids:
    for id in ids:
       location_info = get_location_by_id(id, api_key="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00")
       print("\n监测站点详细信息:")
       print(f"id: {id}")
       print(f"名称: {location_info.get('name', 'N/A')}")
       print(f"城市: {location_info.get('city', 'N/A')}")
       print(f"国家: {location_info.get('country', 'N/A')}")
       print(f"坐标: {location_info.get('coordinates', {}).get('latitude', 'N/A')}, {location_info.get('coordinates', {}).get('longitude', 'N/A')}")
       print(f"参数: {', '.join([p.get('parameter', 'N/A') for p in location_info.get('parameters', [])])}")