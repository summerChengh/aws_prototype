"""
Utility functions for loading and using air quality prediction models
"""

import os
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_ingestion.data_process import AirQualityDataProcessor
from model_utils import perform_feature_engineering
from autogluon.timeseries import TimeSeriesPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
POLLUTANT_MODELS = ["AQI", "O3_aqi", "SO2_aqi", "PM2.5_24h_aqi", "PM10_24h_aqi", "CO_8h_aqi", "NO2_1h_aqi"]
DEFAULT_MODEL_DIR = "./models/automl"
FEATURES = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 
            'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP',
            'FRSHTT', 'pm2.5_24h', 'pm10_24h', 'so2_24h', 
            'o3_8h', 'co_8h', 'o3_1h', 'so2_1h', 'no2_1h',
            'month', 'day_of_year']

def load_model(model_dir: str, pollutant: str):
    """
    Load a trained model for a specific pollutant
    
    Args:
        model_dir: Directory containing the models
        pollutant: Pollutant name (e.g., "AQI", "PM2.5_24h_aqi")
    
    Returns:
        Loaded model object
    """
    try:
        logger.info(f"Loading model for {pollutant} from {model_dir}")

        # Check if model directory exists
        pollutant_dir = os.path.join(model_dir, pollutant)
        if not os.path.exists(pollutant_dir):
            logger.warning(f"Model directory for {pollutant} not found: {pollutant_dir}")
            # return a dummy model
            return DummyModel(pollutant)

        model = TimeSeriesPredictor.load(pollutant_dir)
        return model

    except Exception as e:
        logger.error(f"Error loading model for {pollutant}: {e}")
        # Return dummy model on error
        return DummyModel(pollutant)


def predict_aqi(model, features: Dict[str, Any]) -> float:
    """
    Make an AQI prediction using the provided model and features
    
    Args:
        model: Trained model object
        features: Dictionary of feature values
    
    Returns:
        Predicted AQI value
    """
    try:
        logger.info("Making AQI prediction")

        # In a real implementation, you would transform features and call the model
        # For example:
        # df = pd.DataFrame([features])
        # prediction = model.predict(df)
        # return prediction[0]

        # For demonstration, use the dummy model's predict method
        return model.predict(features)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        # Return a default value on error
        return 75.0


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

        # In a real implementation, you would retrieve data from a database or API
        # For demonstration, return dummy data

        start_date = end_date - timedelta(days = (days - 1))

        processor = AirQualityDataProcessor(start_date, end_date, api_key=api_key)

        df = processor.run_full_pipeline([city_id])

        df_feature = perform_feature_engineering(df)

        return df

    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        # Return empty DataFrame on error
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
