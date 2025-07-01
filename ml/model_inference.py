"""
Utility functions for loading and using air quality prediction models
"""

import os, sys
import logging
from typing import Dict, Any, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_ingestion.data_process import AirQualityDataProcessor
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_historical_data(city_id: str, end_date: str = None, days: int = 14, api_key=None) -> pd.DataFrame:
    """
    Get historical data for a city for the specified number of days

    Args:
        city_id: City identifier
        end_date: End date in format "YYYY-MM-DD"
        days: Number of days of history to retrieve

    Returns:
        DataFrame with historical data
    """
    try:
        if end_date is None:
            end_date_obj = datetime.now()
            start_date_obj = end_date_obj - timedelta(days=(days - 1)) 
        else:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        logger.info(f"Getting {days} days of historical data for {city_id} ending at {end_date}")

        start_date_obj = end_date_obj - timedelta(days=(days - 1))
        start_date = start_date_obj.strftime("%Y-%m-%d")
        end_date = end_date_obj.strftime("%Y-%m-%d")
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
        
    def load_model(self) -> 'DeployedModelInference':
        """
        加载指定模型
            
        Returns:
            self: 返回实例本身，支持链式调用
        """
        
        if not os.path.exists(self.base_deploy_dir):
            raise FileNotFoundError(f"No deployed model found for {self.base_deploy_dir}")
        
        metadata_path = os.path.join(self.base_deploy_dir, "metadata.json")
        feature_config_path = os.path.join(self.base_deploy_dir, "feature_config.joblib")
        
        # 加载元数据
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            self.metadata = {}
        
        # 从元数据获取模型基础信息
        model_name = self.metadata["model_name"]
        self.target_col = self.metadata["target_col"]
        self.version = self.metadata["version"]
        self.model_dir = os.path.join(self.base_deploy_dir, self.target_col, self.version)

        # 检查必要文件是否存在
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # 加载模型
        logger.info(f"Loading deployed model for {self.target_col}, version {self.version} from {self.model_dir}")
        self.predictor = TimeSeriesPredictor.load(self.model_dir)

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
    
    def prepare_data(self, input_df: pd.DataFrame, date=None) -> TimeSeriesDataFrame:
        """
        准备用于预测的数据，包括预处理、去重、补全历史数据等步骤
        
        Args:
            input_df: 输入的历史数据DataFrame，需包含STATION和date等特征
            date: 可选，预测的起始日期，如果提供则会补全历史数据
            
        Returns:
            处理后的TimeSeriesDataFrame
        """
        # 1. 基础预处理
        df = self._preprocess_input_data(input_df)
        
        # 2. 去重处理
        df = self._remove_duplicates(df)
        
        # 3. 如果提供了日期，补全历史数据
        if date is not None:
            df = self._complete_historical_data(df, date)
            
            # 4. 构建预测数据
            ts_df = self._create_time_series_df(df)
            
            # 5. 添加未来数据点用于预测
        #    ts_df = self._add_future_data_points(ts_df, date)
        else:
            # 简单转换为TimeSeriesDataFrame
            ts_df = self._create_time_series_df(df)
        
        return ts_df
    
    def _preprocess_input_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理输入数据，包括日期转换、特征检查等
        
        Args:
            input_df: 输入的历史数据DataFrame
            
        Returns:
            预处理后的DataFrame
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
            
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        去除重复的(STATION, date)组合，保留缺失值最少的行
        
        Args:
            df: 输入DataFrame
            
        Returns:
            去重后的DataFrame
        """
        # 按照station和date聚合，只保留缺失值最少的行
        group_cols = ['STATION', 'date']
        if df.duplicated(subset=group_cols).any():
            # 计算每行的缺失值数量
            df['_na_count'] = df.isna().sum(axis=1)
            # 对每组，保留缺失值最少的那一行（如有并列，取第一行）
            df = df.sort_values('_na_count').groupby(group_cols, as_index=False).first()
            df = df.drop(columns=['_na_count'])
            logger.info(f"Removed duplicate entries for {len(df)} unique (STATION, date) combinations")
        
        return df
    
    def _complete_historical_data(self, df: pd.DataFrame, date) -> pd.DataFrame:
        """
        补全每个station在指定日期前的历史数据
        
        Args:
            df: 输入DataFrame
            date: 预测的起始日期
            
        Returns:
            补全历史数据后的DataFrame
        """
        # 补齐每个station在date前context_len天的历史数据，缺失则插入
        context_len = self.metadata.get("context_length", 14)
        # end_date取date前一天
        if isinstance(date, str):
            end_date = pd.to_datetime(date) - pd.Timedelta(days=1)
        else:
            end_date = date - pd.Timedelta(days=1)
            
        start_date = end_date - pd.Timedelta(days=context_len - 1)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        logger.info(f"Completing historical data from {start_date} to {end_date}")

        # 先收集每个station的经纬度
        station_latlon = df.groupby("STATION")[["LATITUDE", "LONGITUDE", "ELEVATION"]].first().to_dict("index")

        # 构建所有应有的(station, date)组合
        all_stations = df["STATION"].unique()
        full_index = pd.MultiIndex.from_product([all_stations, date_range], names=["STATION", "date"])
        df = df.set_index(["STATION", "date"])
        df = df.reindex(full_index)

        # 补全LATITUDE/LONGITUDE
        for station, latlon in station_latlon.items():
            mask = df.index.get_level_values("STATION") == station
            for col in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
                df.loc[mask, col] = latlon.get(col)

        return df.reset_index()
    
    def _create_time_series_df(self, df: pd.DataFrame) -> TimeSeriesDataFrame:
        """
        将DataFrame转换为TimeSeriesDataFrame
        
        Args:
            df: 输入DataFrame
            
        Returns:
            TimeSeriesDataFrame对象
        """
        # 重命名为AutoGluon格式
        rename_dict = {'STATION': 'item_id', 'date': 'timestamp'}
        
        # 如果目标列存在，也进行重命名
        if self.target_col in df.columns:
            rename_dict[self.target_col] = "target"
            
        ts_df = df.rename(columns=rename_dict)
        
        # 创建TimeSeriesDataFrame
        try:
            ts_df = TimeSeriesDataFrame.from_data_frame(
                ts_df,
                id_column="item_id",
                timestamp_column="timestamp"
            )
            # 填充缺失值
            ts_df = ts_df.fill_missing_values()
            return ts_df
        except Exception as e:
            logger.error(f"Error creating TimeSeriesDataFrame: {e}")
            raise ValueError(f"Failed to create TimeSeriesDataFrame: {e}")
    
    def _add_future_data_points(self, ts_df: TimeSeriesDataFrame, date) -> TimeSeriesDataFrame:
        """
        添加未来数据点用于预测
        
        Args:
            ts_df: 输入的TimeSeriesDataFrame
            date: 预测的起始日期
            
        Returns:
            添加了未来数据点的TimeSeriesDataFrame
        """
        # 转换为DataFrame以便操作
        past_df = ts_df.reset_index().to_data_frame()
        
        # 为每个站点构建未来的数据点
        future_dfs = []
        prediction_len = self.metadata.get("prediction_length", 1)
        
        # 确保date是datetime对象
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        # 获取所有唯一的站点ID
        item_ids = past_df['item_id'].unique()
        
        for item_id in item_ids:
            # 构建该站点的未来数据点
            item_df = past_df[past_df['item_id'] == item_id]
            future_df = self._build_known_covariates(item_df, date, prediction_len)
            future_dfs.append(future_df)
        
        # 合并所有站点的未来数据
        if future_dfs:
            future_df = pd.concat(future_dfs, ignore_index=True)
            
            # 如果past_df中有target列，在future_df中也添加，但值为None
            if 'target' in past_df.columns:
                future_df['target'] = None
                
            # 合并过去和未来的数据
            combined_df = pd.concat([past_df, future_df], ignore_index=True)
            
            # 转换回TimeSeriesDataFrame
            return TimeSeriesDataFrame.from_data_frame(
                combined_df,
                id_column="item_id",
                timestamp_column="timestamp"
            )
        
        return ts_df

    def _build_known_covariates(self, input_df, date, prediction_len):
        """
        构造known_covariates
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)
        start_date = date.strftime('%Y-%m-%d')
        # 判断'timestamp'是否是input_df的列名之一
        if 'timestamp' in input_df.columns:
            latest_date = input_df['timestamp'].max()
            df = input_df[input_df['timestamp'] == latest_date].copy()
            # 如果有多行，取第一行
            if len(df) > 1:
                df = df.iloc[[0]]
        else:
            # 如果没有date列，直接取最后一行
            df = input_df.iloc[[-1]].copy()
            latest_date = df['timestamp'].values[0] if 'timestamp' in df.columns else None

        df = pd.concat([df] * prediction_len, ignore_index=True)
        # 递增date列
        if 'timestamp' in df.columns:
            df['timestamp'] = [pd.to_datetime(start_date) + pd.Timedelta(days=i) for i in range(prediction_len)]
            df['target'] = None
        return df

    def predict(self, input_df: pd.DataFrame, date=None, use_best_model: bool = True) -> pd.DataFrame:
        """
        使用加载的模型对输入数据进行预测
        
        Args:
            input_df: 输入的历史数据DataFrame，需包含STATION和date等特征
            date: 可选，预测的起始日期
            use_best_model: 是否使用最佳模型进行预测
            
        Returns:
            预测结果DataFrame
        """
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # 准备数据
        ts_df = self.prepare_data(input_df, date)
        
        # 获取最佳模型名称
        model = None
        if use_best_model:
            model = self.metadata.get("best_model")
            if model:
                logger.info(f"Using best model: {model}")
        
        # 预测
        try:
            logger.info(f"Making predictions with model {self.metadata.get('model_name')} version {self.version}")
            
            # 准备known_covariates
            known_covariates = self._prepare_known_covariates(ts_df, date)
            
            # 使用known_covariates进行预测
            predictions = self.predictor.predict(
                ts_df, 
                model=model, 
                known_covariates=known_covariates
            )
            logger.info(f"Prediction successful, got {len(predictions)} predictions")
            return predictions["mean"]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
            
    def _prepare_known_covariates(self, ts_df: TimeSeriesDataFrame, date) -> TimeSeriesDataFrame:
        """
        准备用于预测的已知协变量
        
        Args:
            ts_df: 输入的TimeSeriesDataFrame
            date: 预测的起始日期
            
        Returns:
            包含已知协变量的TimeSeriesDataFrame
        """
        if date is None:
            # 如果没有提供日期，使用数据中的最后日期
            last_dates = ts_df.groupby('item_id').timestamp.max()
            if len(last_dates) == 0:
                raise ValueError("Cannot determine prediction start date from empty dataset")
            date = last_dates.min()
            logger.info(f"Using {date} as prediction start date")
        elif isinstance(date, str):
            date = pd.to_datetime(date)
            
        # 获取预测长度
        prediction_len = self.metadata.get("prediction_length", 7)
        
        # 获取所有唯一的站点ID
        item_ids = ts_df.item_ids

        df = ts_df.reset_index().to_data_frame()

        # 为每个站点构建未来的数据点
        future_dfs = []
        
        for item_id in item_ids:
            item_df = df[df['item_id'] == item_id]
            # 构建该站点的未来数据点（不含目标变量）
            future_df = self._build_known_covariates(item_df, date, prediction_len)
            future_dfs.append(future_df)
        
        # 合并所有站点的未来数据
        if not future_dfs:
            logger.warning("No future data points could be created for known_covariates")
            return None
            
        future_df = pd.concat(future_dfs, ignore_index=True)

        # 转换为TimeSeriesDataFrame
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            future_df,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        
        logger.info(f"Prepared known_covariates with shape {known_covariates.shape} for prediction")
        return known_covariates

    def get_context_len(self):
        return self.metadata.get("context_length", 14)


if __name__=='__main__': 
    predictor = DeployedModelInference()
    predictor.load_model()
    context_len = predictor.get_context_len()
    if isinstance(context_len, str) and context_len.isdigit():
        context_len = int(context_len)
    elif not isinstance(context_len, int):
        logger.error(f"error context_len: {context_len}")
        sys.exit(-1)
    
    df = get_historical_data(city_id="72384023155", end_date="2025-07-01", days=context_len, api_key="9b61af0e97dfc16d9b8032bc54dfc62e677518873508c68796b3745ccd19dd00")
    predict_AQI = predictor.predict(df, "2025-07-02", use_best_model=True)
    print(f"predict_AQI: {predict_AQI}")
    