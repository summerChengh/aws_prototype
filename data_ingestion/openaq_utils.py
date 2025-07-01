import requests
from typing import Dict, Optional, Any


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

