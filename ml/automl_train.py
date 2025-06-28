import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = "./data/processed/20240101_20240131"
INPUT_FILE = os.path.join(DATA_DIR, "nooa_openaq_merged.csv")
OUTPUT_DIR = "./models/automl"
RESULTS_DIR = "./data/results/automl"

# Create directories if they don't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# Model parameters
CONTEXT_LENGTH = 14  # Use 14 days of history
PREDICTION_LENGTH = 1  # Predict 1 day ahead
TEST_SIZE = 0.2
RANDOM_STATE = 42
PRESETS = "high_quality"  # Options: "best_quality", "high_quality", "medium_quality", "low_quality"

# Features and targets
STATIC_FEATURES = ['LATITUDE', 'LONGITUDE', 'ELEVATION']
DYNAMIC_FEATURES = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP', 'FRSHTT']
TARGET_POLLUTANTS = ["O3_aqi", "SO2_aqi", "PM2.5_24h_aqi", "PM10_24h_aqi", "CO_8h_aqi", "NO2_1h_aqi", "AQI"]

def load_data(file_path):
    """Load and preprocess the data"""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Convert DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess data for time series modeling"""
    logger.info("Preprocessing data")
    
    # Check for and handle duplicate timestamps
    if df.duplicated(subset=['STATION', 'DATE']).any():
        logger.warning("Found duplicate timestamps, keeping the first occurrence")
        df = df.drop_duplicates(subset=['STATION', 'DATE'], keep='first')
    
    # 剔除AQI列为空的数据
    if 'AQI' in df.columns:
        before_drop = len(df)
        df = df[~df['AQI'].isna()]
        after_drop = len(df)
        logger.info(f"Dropped {before_drop - after_drop} rows with missing AQI values")
    else:
        logger.warning("Column 'AQI' not found in dataframe; no rows dropped for missing AQI")

    # Sort by station and date
    df = df.sort_values(['STATION', 'DATE'])
    
    # Handle missing values in features
    for col in DYNAMIC_FEATURES:
        if df[col].isna().sum() > 0:
            logger.info(f"Filling missing values in {col}")
            # Fill missing values with forward fill first (use previous day's value)
            df[col] = df.groupby('STATION')[col].transform(lambda x: x.fillna(method='ffill'))
            # Then backward fill for any remaining NAs (for the first day)
            df[col] = df.groupby('STATION')[col].transform(lambda x: x.fillna(method='bfill'))
            # If still missing, fill with column median
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    return df

def create_time_series_df(df, target_col):
    """Create TimeSeriesDataFrame for AutoGluon"""
    logger.info(f"Creating TimeSeriesDataFrame for {target_col}")
    
    # Select relevant columns
    ts_df = df[['STATION', 'DATE', target_col] + STATIC_FEATURES + DYNAMIC_FEATURES].copy()
  
    # Rename columns to match AutoGluon expectations
    ts_df = ts_df.rename(columns={
        'STATION': 'item_id',
        'DATE': 'timestamp',
        target_col: 'target'
    })

    print("ts_df columns:", ts_df.columns.tolist())
    
    # Create TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame(ts_df)
    
    return ts_df

def split_data(ts_df):
    """Split data into train and test sets"""
    logger.info("Splitting data into train and test sets")
    
    # 打印TimeSeriesDataFrame的属性，帮助调试
    print("TimeSeriesDataFrame info:")
    print(f"Shape: {ts_df.shape}")
    print(f"Columns: {ts_df.columns}")
    
    # 正确获取item_ids (在TimeSeriesDataFrame中，item_ids是一个属性而不是列)
    try:
        item_ids = ts_df.item_ids
        logger.info(f"Found {len(item_ids)} unique items using ts_df.item_ids")
    except Exception as e:
        # 如果上面的方法失败，尝试转换为常规DataFrame再获取
        logger.warning(f"Error accessing item_ids: {e}, trying alternative method")
        try:
            # 将TimeSeriesDataFrame转换为常规DataFrame
            df = ts_df.to_pandas()
            # 获取MultiIndex的第一级，即item_id
            item_ids = df.index.get_level_values(0).unique()
            logger.info(f"Found {len(item_ids)} unique items using index level 0")
        except Exception as e2:
            logger.error(f"Alternative method also failed: {e2}")
            # 最后的备选方案
            logger.warning("Using all data for training as fallback")
            return ts_df, ts_df.slice_by_timestep(-PREDICTION_LENGTH, None)
    
    # Split item_ids into train and test
    train_items, test_items = train_test_split(item_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # 使用时间点进行分割而不是使用item_id
    # 获取所有时间戳并排序
    all_timestamps = sorted(ts_df.time_index)
    split_idx = int(len(all_timestamps) * (1 - TEST_SIZE))
    split_timestamp = all_timestamps[split_idx]
    
    logger.info(f"Splitting at timestamp: {split_timestamp}")
    
    # 使用正确的方法分割TimeSeriesDataFrame
    train_data = ts_df.loc[:split_timestamp]
    test_data = ts_df.loc[split_timestamp:]
    
    logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    return train_data, test_data

def train_model(train_data, target_col):
    """Train the time series model"""
    logger.info(f"Training model for {target_col}")
    
    # Initialize predictor
    predictor = TimeSeriesPredictor(
        prediction_length=PREDICTION_LENGTH,
        eval_metric='RMSE',  # Root Mean Squared Error
        path=os.path.join(OUTPUT_DIR, target_col),
        target='target',
        known_covariates_names=DYNAMIC_FEATURES,
        static_features=STATIC_FEATURES,
        verbosity=2,
        context_length=CONTEXT_LENGTH
    )
    
    # Fit the model
    predictor.fit(
        train_data,
        presets=PRESETS,
        time_limit=3600,  # 1 hour time limit
        enable_ensemble=True,
        num_val_windows=2,  # Number of validation windows
        hyperparameters={'DeepAR': {}, 'Transformer': {}, 'TemporalFusionTransformer': {}}
    )
    
    return predictor

def evaluate_model(predictor, test_data, target_col):
    """Evaluate the model on test data"""
    logger.info(f"Evaluating model for {target_col}")
    
    # Make predictions
    predictions = predictor.predict(test_data)
    
    # Evaluate performance
    performance = predictor.evaluate(test_data)
    
    # Save performance metrics
    performance_df = pd.DataFrame(performance).T
    performance_df.to_csv(os.path.join(RESULTS_DIR, f"{target_col}_performance.csv"))
    
    logger.info(f"Performance for {target_col}: {performance}")
    
    # Plot predictions for a sample item
    sample_item = test_data.item_id.unique()[0]
    sample_data = test_data.loc[sample_item]
    sample_pred = predictions.loc[sample_item]
    
    plt.figure(figsize=(12, 6))
    plt.plot(sample_data.index, sample_data['target'], label='Actual')
    plt.plot(sample_pred.index, sample_pred['mean'], label='Predicted', color='red')
    plt.fill_between(
        sample_pred.index,
        sample_pred['0.1'], sample_pred['0.9'],
        color='red', alpha=0.3, label='80% Prediction Interval'
    )
    plt.title(f'{target_col} Predictions for {sample_item}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{target_col}_sample_prediction.png"))
    
    return performance

def feature_importance(predictor, target_col):
    """Extract feature importance if available"""
    logger.info(f"Extracting feature importance for {target_col}")
    
    try:
        importance = predictor.feature_importance(summarize=True)
        if importance is not None:
            importance.to_csv(os.path.join(RESULTS_DIR, f"{target_col}_feature_importance.csv"))
            
            # Plot top 10 features
            plt.figure(figsize=(10, 6))
            importance.head(10).plot(kind='barh')
            plt.title(f'Top 10 Features for {target_col}')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"{target_col}_feature_importance.png"))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

def main():
    """Main function to run the training pipeline"""
    start_time = datetime.now()
    logger.info(f"Starting AutoML training pipeline at {start_time}")
    
    # Load and preprocess data
    df = load_data(INPUT_FILE)
    df = preprocess_data(df)

    print("preprocess_data: ")
    print(df[["STATION", "DATE", "AQI"]])

    # Train and evaluate models for each target pollutant
    results = {}
    for pollutant in TARGET_POLLUTANTS:
        logger.info(f"Processing {pollutant}")
        
        # 如果pollutant列缺失值超过一定比例，则跳过下面的步骤
        missing_ratio = df[pollutant].isna().mean()
        max_missing_ratio = 0.91  # 可根据需要调整阈值
        if missing_ratio > max_missing_ratio:
            logger.warning(f"Skipping {pollutant} because missing ratio ({missing_ratio:.2%}) exceeds threshold ({max_missing_ratio:.2%})")
            continue

        ts_df = create_time_series_df(df, pollutant)

        
        # Split data
        train_data, test_data = split_data(ts_df)
        
        # Train model
        predictor = train_model(train_data, pollutant)
        
        # Evaluate model
        performance = evaluate_model(predictor, test_data, pollutant)
        results[pollutant] = performance
        
        # Get feature importance
        feature_importance(predictor, pollutant)
        
        # Save leaderboard
        lb = predictor.leaderboard()
        lb.to_csv(os.path.join(RESULTS_DIR, f"{pollutant}_leaderboard.csv"))
        
        # Save model info
        with open(os.path.join(RESULTS_DIR, f"{pollutant}_model_info.txt"), 'w') as f:
            f.write(f"Model path: {os.path.join(OUTPUT_DIR, pollutant)}\n")
            f.write(f"Best model: {predictor.get_model_best()}\n")
            f.write(f"All models: {predictor.get_model_names()}\n")
    
    # Summarize results
    summary = pd.DataFrame(results).T
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_performance.csv"))
    
    end_time = datetime.now()
    logger.info(f"AutoML training pipeline completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
