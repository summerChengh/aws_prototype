"""
FastAPI service for air quality prediction model inference
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field
import uvicorn

# 导入模型推理相关函数
from ml.model_inference import (
    load_model, predict_aqi, get_historical_data, 
    calculate_trend, POLLUTANT_MODELS, DEFAULT_MODEL_DIR
)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 初始化FastAPI应用
app = FastAPI(
    title="Air Quality Prediction API",
    description="API for predicting air quality metrics using trained models",
    version="1.0.0"
)

# 预加载模型
models = {}

@app.on_event("startup")
async def startup_load_models():
    """启动时加载所有模型"""
    global models
    logger.info("Loading models on startup")
    for pollutant in POLLUTANT_MODELS:
        models[pollutant] = load_model(DEFAULT_MODEL_DIR, pollutant)
    logger.info(f"Loaded {len(models)} models")

# 定义API数据模型
class PredictionRequest(BaseModel):
    city_id: str = Field(..., description="City identifier")
    date: Optional[str] = Field(None, description="Prediction date (YYYY-MM-DD), defaults to tomorrow")
    features: Optional[Dict[str, float]] = Field(None, description="Additional feature values")
    pollutants: Optional[List[str]] = Field(None, description="Specific pollutants to predict")

class PredictionResponse(BaseModel):
    city_id: str
    prediction_date: str
    predictions: Dict[str, float]
    trends: Dict[str, float]
    confidence: Dict[str, Dict[str, float]]
    
class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    available_pollutants: List[str]
    timestamp: str

   

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测空气质量指标"""
    try:
        # 确定预测日期
        if request.date:
            prediction_date = datetime.strptime(request.date, "%Y-%m-%d")
        else:
            # 默认预测明天
            prediction_date = datetime.now() + timedelta(days=1)
        
        # 获取历史数据
        historical_data = get_historical_data(
            city_id=request.city_id,
            end_date=prediction_date - timedelta(days=1),  # 使用到预测日期前一天的数据
            days=14  # 使用14天的历史数据
        )
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for city {request.city_id}")
        
        # 确定要预测的污染物
        pollutants_to_predict = request.pollutants or POLLUTANT_MODELS
        
        # 存储预测结果
        predictions = {}
        trends = {}
        confidence = {}
        
        # 为每个污染物进行预测
        for pollutant in pollutants_to_predict:
            if pollutant not in models:
                logger.warning(f"Model for {pollutant} not found, skipping")
                continue
                
            # 准备特征
            features = {}
            if request.features:
                features.update(request.features)
                
            # 从历史数据中提取额外特征
            if not historical_data.empty:
                # 添加最近的气象数据
                latest_data = historical_data.iloc[-1].to_dict()
                for feature in ['TEMP', 'DEWP', 'SLP', 'VISIB', 'WDSP']:
                    if feature in latest_data and feature not in features:
                        features[feature] = latest_data[feature]
                
                # 添加日期相关特征
                features['DAY_OF_YEAR'] = prediction_date.timetuple().tm_yday
                features['MONTH'] = prediction_date.month
                
                # 计算趋势
                if pollutant in historical_data.columns:
                    trends[pollutant] = calculate_trend(historical_data[pollutant])
            
            # 进行预测
            model = models[pollutant]
            predicted_value = predict_aqi(model, features)
            predictions[pollutant] = float(predicted_value)
            
            # 添加置信区间（模拟值，实际应从模型获取）
            confidence[pollutant] = {
                "lower_bound": max(0, predicted_value * 0.85),
                "upper_bound": min(500, predicted_value * 1.15)
            }
        
        # 构建响应
        response = {
            "city_id": request.city_id,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
            "predictions": predictions,
            "trends": trends,
            "confidence": confidence
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    return {
        "available_models": list(models.keys()),
        "model_dir": DEFAULT_MODEL_DIR
    }

@app.get("/cities")
async def list_cities(limit: int = Query(10, ge=1, le=100)):
    """列出支持的城市"""
    # 这里应该从数据库或配置或服务接口获取城市列表
    # 为演示返回一些示例城市
    sample_cities = [
        {"id": "BJ", "name": "Beijing", "country": "China"},
        {"id": "SH", "name": "Shanghai", "country": "China"},
        {"id": "NY", "name": "New York", "country": "USA"},
        {"id": "LD", "name": "London", "country": "UK"},
        {"id": "TK", "name": "Tokyo", "country": "Japan"}
    ]
    return {"cities": sample_cities[:limit]}



if __name__ == "__main__":
    # 运行服务
    uvicorn.run(app, host="0.0.0.0", port=8000)
