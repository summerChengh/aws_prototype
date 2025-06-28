import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor
from data_ingestion.cal_aqi import AQICalculator 


calculator = AQICalculator()

df["AQI"] = df.apply(lambda row: calculator.calculate_overall_aqi({
    "PM2.5": row["PM2.5"],
    "PM10": row["PM10"], 
    "O3_8h": row["O3_8h"],
    "O3_1h": row["O3_1h"],
    "CO": row["CO"],
    "NO2": row["NO2"],
    "SO2": row["SO2"]
}), axis=1)


context_length = 14
feature_cols = ["O3", "NO2", "SO2", "dayofyear"]
target_pollutants = ["PM2.5", "PM10", "O3_8h", "CO", "NO2", "SO2"]

# 为每个污染物训练模型
for pollutant in target_pollutants:
    label = pollutant
    train_data = TabularDataset(train_df[feature_cols + [label]])
    predictor = TimeSeriesPredictor(
    prediction_length=1,
    target=label,
    eval_metric='MAE', # 
    context_length=context_length,  # 明确指定用过去14天预测
    presets="high_quality"  # 使用更强模型
    )

    # Todo:  缺失值填充

    predictor.fit(
        train_data,
        known_covariates_names=["temp", "wind", "humidity"]
    )

    # save model



