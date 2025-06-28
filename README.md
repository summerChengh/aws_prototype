# AWS Air Quality Predictor

基于AWS云服务的全球城市空气质量预测系统，集成OpenAQ和NOAA数据，提供高精度空气质量预测和可视化服务。

## 项目概述

本项目利用AWS云服务构建了一个完整的空气质量预测系统，通过整合OpenAQ空气质量数据和NOAA气象数据，结合先进的机器学习技术，为用户提供准确的空气质量预测和个性化的可视化内容。

### 核心功能

- **多源数据整合**: 自动采集和处理来自OpenAQ和NOAA的大规模数据
- **高精度预测**: 利用AutoML技术训练和优化空气质量预测模型
- **时空分析**: 提供基于地理位置和时间维度的空气质量分析
- **个性化内容**: 使用生成式AI创建与城市和空气质量相关的图片
- **API服务**: 提供RESTful API接口供第三方应用集成

## 系统架构

系统采用AWS无服务器架构，包括以下主要组件：

```
                                     ┌───────────────────┐
                                     │                   │
                                     │  NOAA Data (S3)   │
                                     │                   │
                                     └─────────┬─────────┘
                                               │
                                               ▼
┌───────────────────┐              ┌───────────────────────┐              ┌───────────────────┐
│                   │              │                       │              │                   │
│   OpenAQ Data     │──────────────▶    AWS Glue ETL      │◀─────────────│  AWS Lambda       │
│                   │              │                       │              │  (API Fetcher)    │
└───────────────────┘              └───────────┬───────────┘              └───────────────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐              ┌───────────────────┐
                                     │                   │              │                   │
                                     │  S3 Data Lake     │◀─────────────│  AWS Athena       │
                                     │                   │              │  (Query)          │
                                     └─────────┬─────────┘              └───────────────────┘
                                               │
                                               ▼
┌───────────────────┐              ┌───────────────────────┐              ┌───────────────────┐
│                   │              │                       │              │                   │
│  SageMaker        │◀─────────────│  SageMaker Processing │◀─────────────│  SageMaker        │
│  Autopilot        │              │  (Feature Eng.)       │              │  Studio (IDE)     │
│                   │              │                       │              │                   │
└─────────┬─────────┘              └───────────────────────┘              └───────────────────┘
          │
          ▼
┌───────────────────┐              ┌───────────────────────┐              ┌───────────────────┐
│                   │              │                       │              │                   │
│  SageMaker        │──────────────▶    Lambda             │──────────────▶  Amazon Bedrock   │
│  Model Registry   │              │    (Inference)        │              │  (Image Gen)      │
│                   │              │                       │              │                   │
└───────────────────┘              └───────────┬───────────┘              └───────────────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐              ┌───────────────────┐
                                     │                   │              │                   │
                                     │  API Gateway      │◀─────────────│  CloudFront CDN   │
                                     │                   │              │                   │
                                     └─────────┬─────────┘              └───────────────────┘
                                               │
                                               ▼
                                     ┌───────────────────┐
                                     │                   │
                                     │  End Users        │
                                     │                   │
                                     └───────────────────┘
```

详细架构设计请参阅 [架构文档](docs/DESIGN.md)。

## 技术栈

- **AWS云服务**:
  - Amazon S3: 数据湖存储
  - AWS Glue: ETL数据处理
  - Amazon SageMaker: 机器学习模型开发和部署
  - AWS Lambda: 无服务器计算
  - Amazon API Gateway: API管理
  - Amazon Bedrock: 生成式AI服务
  
- **开发技术**:
  - Python: 主要开发语言
  - PySpark: 大规模数据处理
  - AutoGluon: 自动机器学习框架
  - React: 前端开发

## 快速开始

### 前提条件

- AWS账户
- Python 3.8+
- AWS CLI已配置

### 安装步骤

1. 克隆仓库:
   ```bash
   git clone https://github.com/yourusername/aws-air-quality-predictor.git
   cd aws-air-quality-predictor
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

3. 配置AWS凭证:
   ```bash
   aws configure
   ```

4. 部署基础设施:
   ```bash
   cd infrastructure
   ./deploy.sh
   ```

## 项目结构

```
aws-air-quality-predictor/
├── data_ingestion/           # 数据摄取模块
│   ├── fetch_openaq.py       # OpenAQ数据获取
│   └── fetch_noaa.py         # NOAA数据获取
├── data_processing/          # 数据处理模块
│   ├── etl_jobs/             # ETL作业
│   └── feature_engineering/  # 特征工程
├── ml/                       # 机器学习模块
│   ├── models/               # 模型定义
│   ├── training/             # 训练脚本
│   └── evaluation/           # 评估脚本
├── api/                      # API服务
│   ├── endpoints/            # API端点定义
│   └── middleware/           # 中间件
├── frontend/                 # 前端应用
├── infrastructure/           # 基础设施代码
│   ├── cloudformation/       # CloudFormation模板
│   └── terraform/            # Terraform配置
├── docs/                     # 文档
│   └── DESIGN.md             # 架构设计文档
├── tests/                    # 测试
└── README.md                 # 项目说明
```

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

项目维护者 - [@yourusername](https://github.com/yourusername)

项目链接: [https://github.com/yourusername/aws-air-quality-predictor](https://github.com/yourusername/aws-air-quality-predictor) 