import pandas as pd
import numpy as np
import os
from datetime import datetime
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AirQualityPredictor:
    def __init__(self,
                 data_dir="./data/processed/20160101_20220608",
                 input_file=None,
                 output_dir=None,
                 results_dir=None,
                 deploy_dir=None,
                 context_length=14,
                 prediction_length=1,
                 presets="high_quality",
                 default_hyperparameters=None):
        """初始化空气质量预测器"""
        # 配置路径
        self.data_dir = data_dir
        self.input_file = input_file or os.path.join(data_dir, "nooa_openaq_merged.csv")
        # 使用当前时间戳到秒作为模型版本
        self.model_version = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = output_dir or f"./models/automl/{self.model_version}/{os.path.basename(data_dir)}"
        self.results_dir = results_dir or f"./data/results/automl/{self.model_version}/{os.path.basename(data_dir)}/"
        self.deploy_dir = deploy_dir or "./models/deploy"

        # 创建目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        # 模型参数
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.presets = presets
        self.random_state = 42

        # 特征和目标
        self.static_features = ['LATITUDE', 'LONGITUDE', 'ELEVATION']
        self.dynamic_features = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP',
                                 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP',
                                 'FRSHTT', 'pm25_24h', 'pm10_24h', 'so2_24h',
                                 'o3_8h', 'co_8h', 'o3_1h', 'so2_1h', 'no2_1h']
        self.target_pollutants = ["AQI"]
        #        self.target_pollutants = ["AQI", "O3_aqi", "SO2_aqi", "PM2.5_24h_aqi",
        #                                 "PM10_24h_aqi", "CO_8h_aqi", "NO2_1h_aqi"]

        # 数据
        self.df = None
        self.results = {}

        self.default_hyperparameters = {
            'DirectTabular': {},
            'DeepAR': {},
            'AutoARIMA': {},
            'TemporalFusionTransformer': {}
        }
        # 默认超参数配置
        if default_hyperparameters is not None:
            self.default_hyperparameters = default_hyperparameters

    def load_data(self):
        """加载数据"""
        logger.info(f"Loading data from {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file)
            self.df['date'] = pd.to_datetime(self.df['date'])
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_time_series_df(self, df, target_col):
        """创建时间序列数据框"""
        logger.info(f"Creating TimeSeriesDataFrame for {target_col}")

        ts_df = df[['STATION', 'date', target_col] + self.static_features + self.dynamic_features].copy()
        ts_df = ts_df.rename(columns={
            'STATION': 'item_id',
            'date': 'timestamp',
            target_col: 'target'
        })

        return TimeSeriesDataFrame(ts_df)

    def train_model(self, train_data, target_col, hyperparameters=None):
        """训练模型
        
        Args:
            train_data: 训练数据
            target_col: 目标列名
            hyperparameters: 可选，模型超参数，若不提供则使用默认配置
        """
        logger.info(f"Training model for {target_col}")

        # 使用提供的超参数或默认超参数
        if hyperparameters is not None:
            model_hyperparameters = hyperparameters
        else:
            model_hyperparameters = self.default_hyperparameters

        predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            eval_metric='RMSE',
            path=os.path.join(self.output_dir, target_col),
            target='target',
            known_covariates_names=self.dynamic_features,
            verbosity=2,
            freq='D'
        )

        predictor.fit(
            train_data,
            presets=self.presets,
            time_limit=3600,
            enable_ensemble=True,
            num_val_windows=2,
            hyperparameters=model_hyperparameters
        )

        return predictor

    def evaluate_model(self, predictor, data, target_col):
        """评估模型"""
        logger.info(f"Evaluating model for {target_col}")
        try:
            if hasattr(predictor, "model_names"):
                all_performance = {}
                for model_name in predictor.model_names():
                    logger.info(f"Evaluating sub-model: {model_name} for {target_col}")
                    performance = predictor.evaluate(data, model=model_name, metrics=["RMSE", "MAE", "MSE"])
                    all_performance[model_name] = performance
                # 保存所有子模型的评估结果
                all_perf_df = pd.DataFrame(all_performance).T
                all_perf_df.to_csv(os.path.join(self.results_dir, f"{target_col}_all_models_performance.csv"))
                return all_performance
            else:
                performance = predictor.evaluate(data, metrics=["RMSE", "MAE", "MSE"])
                performance_df = pd.DataFrame([{'score': performance}])
                performance_df.to_csv(os.path.join(self.results_dir, f"{target_col}_performance.csv"))
                logger.info("predictor does not support model_names(), only evaluating best model.")
                return performance
        except Exception as e:
            logger.warning(f"Error evaluating models for {target_col}: {e}")

        return None

    def feature_importance(self, predictor, target_col):
        """提取特征重要性"""
        logger.info(f"Extracting feature importance for {target_col}")

        try:
            # 动态计算合适的subsample_size
            if hasattr(predictor, 'train_data') and predictor.train_data is not None:
                item_count = len(predictor.train_data.item_id.unique())
                # 确保subsample_size不超过数据中的项目数
                subsample_size = max(1, min(10, item_count - 1))
                logger.info(f"Using subsample_size={subsample_size} for feature importance")

                importance = predictor.feature_importance(subsample_size=subsample_size)
            else:
                # 如果无法确定项目数，不指定subsample_size
                importance = predictor.feature_importance()

            if importance is not None:
                importance.to_csv(os.path.join(self.results_dir, f"{target_col}_feature_importance.csv"))

                plt.figure(figsize=(10, 6))
                importance.head(10).plot(kind='barh')
                plt.title(f'Top 10 Features for {target_col}')
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, f"{target_col}_feature_importance.png"))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

    def train_all_models(self, model_configs=None):
        """训练所有模型
        
        Args:
            model_configs: 可选，{污染物名称: 超参数配置} 的字典
        """

        model_configs = model_configs or {}

        start_time = datetime.now()
        logger.info(f"Starting AutoML training pipeline at {start_time}")

        # 加载数据
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # 处理每个污染物
        for target in self.target_pollutants:
            logger.info(f"Processing {target}")

            # 过滤target列为空的数据
            if target in df.columns:
                before_drop = len(df)
                df = df[~df[target].isna()]
                after_drop = len(df)
                logger.info(f"Dropped {before_drop - after_drop} rows with missing {target} values")

            # 保证target为dtype
            if not pd.api.types.is_numeric_dtype(df[target]):
                df[target] = pd.to_numeric(df[target], errors='coerce')

            # 按站点和日期排序
            df = df.sort_values(['STATION', 'date'])

            df = self.replace_with_previous_day(df, self.dynamic_features)

            # 创建时间序列数据
            ts_df = self.create_time_series_df(df, target)

            # 填充缺失值 
            ts_df = ts_df.fill_missing_values()

            # 划分数据
            train_data, eval_data = ts_df.train_test_split(self.prediction_length)

            # 获取针对该污染物的特定超参数，如果有的话
            pollutant_hyperparams = model_configs.get(target, None)

            # 训练模型时传入特定超参数
            predictor = self.train_model(train_data, target, hyperparameters=pollutant_hyperparams)

            # 评估模型
            performance = self.evaluate_model(predictor, eval_data, target)
            if performance is not None:
                self.results[target] = performance
                # debug
                print(f"performance: {performance}")

            # 特征重要性
            self.feature_importance(predictor, target)

            # 保存模型信息
            self._save_model_info(predictor, target)

            self.save_best_model_for_deployment(predictor, target_col=target, performance=performance)

        # 汇总结果
        self._summarize_results()

        end_time = datetime.now()
        logger.info(f"AutoML training pipeline completed in {end_time - start_time}")

        return self.results

    def _save_model_info(self, predictor, target_col):
        """保存模型信息"""
        # 保存leaderboard
        lb = predictor.leaderboard()
        lb.to_csv(os.path.join(self.results_dir, f"{target_col}_leaderboard.csv"))

        # 保存模型信息
        with open(os.path.join(self.results_dir, f"{target_col}_model_info.txt"), 'w') as f:
            f.write(f"Model path: {os.path.join(self.output_dir, target_col)}\n")
            f.write(f"Best model: {predictor.model_best}\n")
            f.write(f"All models: {predictor.model_names}\n")

    def _summarize_results(self):
        """汇总结果"""
        if self.results:
            summary = pd.DataFrame(self.results).T
            summary.to_csv(os.path.join(self.results_dir, "summary_performance.csv"))

    def save_best_model_for_deployment(self, predictor, target_col, performance={}):
        """保存最优模型用于线上部署
        """
        # 创建部署目录
        deploy_dir = self.deploy_dir
        deploy_model_dir = os.path.join(deploy_dir, target_col, self.model_version)
        Path(deploy_dir).mkdir(parents=True, exist_ok=True)
        Path(deploy_model_dir).mkdir(parents=True, exist_ok=True)

        try:
            model_path = os.path.join(self.output_dir, target_col)
            if not os.path.exists(model_path):
                logger.error(f"模型 {target_col} 不存在，请先训练模型")
                return False

            # 1. 将model_path下的所有文件和子目录拷贝到deploy_model_dir
            if os.path.exists(deploy_model_dir):
                for root, dirs, files in os.walk(deploy_model_dir, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"无法删除文件 {file_path}: {e}")
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        try:
                            shutil.rmtree(dir_path)
                        except Exception as e:
                            logger.warning(f"无法删除目录 {dir_path}: {e}")
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(deploy_model_dir, item)
                if item == "logs":
                    continue
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

            # 2. 保存模型元数据
            metadata = {
                "model_name": target_col,
                "version": self.model_version,
                "target_col": target_col,
                "created_at": datetime.now().isoformat(),
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "freq": "D",
                "features": {
                    "static": self.static_features,
                    "dynamic": self.dynamic_features
                },
                "best_model": predictor.model_best,
                "performance": performance
            }

            with open(os.path.join(deploy_dir, "metadata.json"), "w") as f:
                import json
                json.dump(metadata, f, indent=2)

            # 3. 保存特征处理逻辑
            import joblib
            joblib.dump({
                "static_features": self.static_features,
                "dynamic_features": self.dynamic_features
            }, os.path.join(deploy_dir, "feature_config.joblib"))

            logger.info(f"成功保存模型 {target_col} 到 {deploy_dir}")
            return True

        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False

    def replace_with_previous_day(self, df, columns):
        """
        将df中的指定特征列的值替换为前一天的值（按STATION分组，按date排序，shift(1)）。
        Args:
            df: 输入DataFrame，必须包含'STATION'和'date'列
            columns: 需要替换的特征列名列表
        Returns:
            替换后的DataFrame
        """
        for col in columns:
            if col in df.columns:
                df[col] = df.groupby('STATION')[col].shift(1)
        return df


if __name__ == "__main__":
    # 使用默认超参数
    predictor = AirQualityPredictor()
    '''
    # 自定义参数
    predictor = AirQualityPredictor(
        data_dir="./data/processed/custom_dataset",
        presets="medium_quality",
        prediction_length=3
    )
    predictor.train_all_models()
   '''
    # 为不同污染物配置不同模型
    custom_configs = {
        "AQI": {'DirectTabular': {}, 'DeepAR': {}},
    }

    predictor.train_all_models(model_configs=custom_configs)
