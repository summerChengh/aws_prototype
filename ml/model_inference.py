"""
Utility functions for loading and using air quality prediction models
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
from data_ingestion.data_process import AirQualityDataProcessor
from model_utils import perform_feature_engineering
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_historical_data(city_id: str, end_date=datetime.now(), days: int = 14, api_key=None) -> pd.DataFrame:
    """
    Get historical data for a city for the specified number of days
    
    Args:
        city_id: City identifier
        days: Number of days of history to retrieve
    
    Returns:
        DataFrame with historical data
    """
    try:
        logger.info(f"Getting {days} days of historical data for {city_id}")

        start_date = end_date - timedelta(days = (days - 1))

        processor = AirQualityDataProcessor(start_date, end_date, api_key=api_key)

        df = processor.run_full_pipeline([city_id])

        return df

    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        return pd.DataFrame()


def calculate_trend(series: pd.Series) -> float:
    """
    Calculate the trend of a time series
    
    Args:
        series: Pandas Series with time series data
    
    Returns:
        Trend value (positive for increasing, negative for decreasing)
    """
    try:
        # Simple linear trend calculation
        if len(series) < 2:
            return 0

        x = np.arange(len(series))
        y = series.values

        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]

        return slope

    except Exception as e:
        logger.error(f"Error calculating trend: {e}")
        return 0.0


class DummyModel:
    """
    Dummy model class for demonstration purposes
    """

    def __init__(self, pollutant):
        self.pollutant = pollutant
        logger.info(f"Initialized dummy model for {pollutant}")

    @staticmethod
    def predict(features):
        """
        Make a dummy prediction
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Predicted value
        """
        # Generate a somewhat realistic AQI value
        base_value = 75  # Moderate AQI as base

        # Add some variation based on features if available
        if isinstance(features, dict):
            # Temperature effect: higher temps tend to increase AQI
            temp_effect = features.get('TEMP', 25) - 20  # Effect relative to 20°C

            # Wind speed effect: higher wind speeds tend to decrease AQI
            wind_effect = -features.get('WDSP', 5) / 2

            # Day of year effect: summer months tend to have higher AQI
            day_of_year = features.get('DAY_OF_YEAR', datetime.now().timetuple().tm_yday)
            seasonal_effect = 15 * np.sin((day_of_year - 172) / 365 * 2 * np.pi)  # Peak in summer

            # Combine effects
            aqi = base_value + temp_effect + wind_effect + seasonal_effect

            # Add random variation
            aqi += np.random.normal(0, 10)

            # Ensure AQI is within valid range
            aqi = max(0, min(500, aqi))

            return aqi

        # If features not provided as dict, return base value with some randomness
        return base_value + np.random.normal(0, 15)


class DeployedModelInference:
    """
    用于加载和使用已部署模型进行推理的类
    """
    
    def __init__(self, base_deploy_dir: str = "./models/deploy"):
        """
        初始化模型推理类
        
        Args:
            base_deploy_dir: 模型部署的基础目录
        """
        self.base_deploy_dir = base_deploy_dir
        self.predictor = None
        self.metadata = {}
        self.feature_config = {}
        self.target_col = None
        self.version = None
        self.model_dir = None
        
    def load_model(self, target_col: str, version: str = None) -> 'DeployedModelInference':
        """
        加载指定目标的模型，version为None时自动选择最新版本
        
        Args:
            target_col: 目标污染物名称
            version: 可选，模型版本号
            
        Returns:
            self: 返回实例本身，支持链式调用
        """
        self.target_col = target_col
        deploy_base = os.path.join(self.base_deploy_dir, target_col)
        
        if not os.path.exists(deploy_base):
            raise FileNotFoundError(f"No deployed model found for {target_col}")
        
        # 获取所有版本并按时间排序
        versions = sorted(os.listdir(deploy_base), reverse=True)
        if not versions:
            raise FileNotFoundError(f"No version found for {target_col} in deploy directory")
        
        # 如果未指定版本，使用最新版本
        if version is None:
            version = versions[0]
        
        self.version = version
        self.model_dir = os.path.join(deploy_base, version)
        model_path = os.path.join(self.model_dir, "model")
        metadata_path = os.path.join(self.model_dir, "metadata.json")
        feature_config_path = os.path.join(self.model_dir, "feature_config.joblib")
        
        # 检查必要文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # 加载模型
        logger.info(f"Loading deployed model for {target_col}, version {version} from {self.model_dir}")
        self.predictor = TimeSeriesPredictor.load(model_path)
        
        # 加载元数据
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = {}
        
        # 加载特征配置
        if os.path.exists(feature_config_path):
            self.feature_config = joblib.load(feature_config_path)
        else:
            logger.warning(f"Feature config file not found: {feature_config_path}")
            self.feature_config = {"static_features": [], "dynamic_features": []}
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        if not self.metadata:
            return {
                "target_col": self.target_col,
                "version": self.version,
                "model_dir": self.model_dir
            }
        
        return {
            "target_col": self.target_col,
            "version": self.version,
            "model_dir": self.model_dir,
            "model_name": self.metadata.get("model_name"),
            "created_at": self.metadata.get("created_at"),
            "prediction_length": self.metadata.get("prediction_length"),
            "freq": self.metadata.get("freq"),
            "best_model": self.metadata.get("best_model"),
            "features": self.feature_config
        }
    
    def get_feature_requirements(self) -> Tuple[List[str], List[str]]:
        """
        获取模型所需的特征列表
        
        Returns:
            (静态特征列表, 动态特征列表)
        """
        static_features = self.feature_config.get("static_features", [])
        dynamic_features = self.feature_config.get("dynamic_features", [])
        return static_features, dynamic_features
    
    def prepare_data(self, input_df: pd.DataFrame) -> TimeSeriesDataFrame:
        """
        准备用于预测的数据
        
        Args:
            input_df: 输入的历史数据DataFrame，需包含STATION和date等特征
            
        Returns:
            处理后的TimeSeriesDataFrame
        """
        # 获取静态和动态特征
        static_features, dynamic_features = self.get_feature_requirements()
        
        # 预处理输入数据
        df = input_df.copy()
        
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 按站点和日期排序
        df = df.sort_values(['STATION', 'date'])
        
        # 检查必要的特征是否存在
        missing_features = []
        for feature in static_features + dynamic_features:
            if feature not in df.columns:
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Missing features in input data: {missing_features}")
        
        # 重命名为AutoGluon格式
        ts_df = df.rename(columns={'STATION': 'item_id', 'date': 'timestamp'})
        
        # 创建TimeSeriesDataFrame
        try:
            ts_df = TimeSeriesDataFrame(ts_df)
        except Exception as e:
            logger.error(f"Error creating TimeSeriesDataFrame: {e}")
            raise ValueError(f"Failed to create TimeSeriesDataFrame: {e}")
        
        # 填充缺失值
        ts_df = ts_df.fill_missing_values()
        
        return ts_df
    
    def predict(self, input_df: pd.DataFrame, use_best_model: bool = True) -> pd.DataFrame:
        """
        使用加载的模型对输入数据进行预测
        
        Args:
            input_df: 输入的历史数据DataFrame，需包含STATION和date等特征
            use_best_model: 是否使用最佳模型进行预测
            
        Returns:
            预测结果DataFrame
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 准备数据
        ts_df = self.prepare_data(input_df)
        
        # 获取最佳模型名称
        model = None
        if use_best_model:
            model = self.metadata.get("best_model")
            if model:
                logger.info(f"Using best model: {model}")
        
        # 预测
        try:
            logger.info(f"Making predictions with model {self.metadata.get('model_name')} version {self.version}")
            predictions = self.predictor.predict(ts_df, model=model)
            logger.info(f"Prediction successful, got {len(predictions)} predictions")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")


# 为了向后兼容，保留原来的函数接口
def load_deployed_model(target_col: str, version: str = None):
    """
    从./models/deploy加载指定目标的模型，version为None时自动选择最新版本。
    返回TimeSeriesPredictor对象和模型元数据。
    
    Args:
        target_col: 目标污染物名称
        version: 可选，模型版本号
    
    Returns:
        tuple: (predictor, metadata, feature_config)
    """
    model_inference = DeployedModelInference()
    model_inference.load_model(target_col, version)
    return model_inference.predictor, model_inference.metadata, model_inference.feature_config


def predict_with_deployed_model(target_col: str, input_df: pd.DataFrame, version: str = None):
    """
    使用部署的模型对输入数据进行预测。
    
    Args:
        target_col: 目标污染物名称
        input_df: 输入的历史数据DataFrame，需包含STATION和date等特征
        version: 可选，模型版本号
    
    Returns:
        预测结果DataFrame
    """
    model_inference = DeployedModelInference()
    model_inference.load_model(target_col, version)
    return model_inference.predict(input_df)
