# 空气质量预测系统 - 前端

本项目是空气质量预测系统的前端部分，基于Vue 3 + TypeScript + Element Plus开发。

## 功能特点

- 城市选择：支持选择不同城市进行空气质量预测
- 日期选择：选择未来日期进行预测
- 预测结果展示：展示AQI指数、污染等级和主要污染物数据
- 图像可视化：展示AI生成的空气质量可视化图像
- 健康建议：根据预测结果提供相应的健康建议

## 技术栈

- Vue 3：前端框架
- TypeScript：类型系统
- Vite：构建工具
- Pinia：状态管理
- Element Plus：UI组件库
- Axios：HTTP请求

## 项目结构

```
frontend/
├── public/              # 静态资源
├── src/
│   ├── assets/          # 资源文件
│   ├── components/      # 公共组件
│   ├── router/          # 路由配置
│   ├── stores/          # Pinia状态管理
│   ├── views/           # 页面视图
│   ├── App.vue          # 根组件
│   └── main.ts          # 入口文件
├── package.json         # 项目依赖
├── tsconfig.json        # TypeScript配置
└── vite.config.ts       # Vite配置
```

## 开发指南

### 安装依赖

```bash
npm install
```

### 启动开发服务器

```bash
npm run dev
```

### 构建生产版本

```bash
npm run build
```

### 预览生产版本

```bash
npm run preview
```

## API接口

### 获取城市列表

```
GET /api/cities
```

### 预测空气质量

```
POST /api/predict
{
  "city_id": "beijing",
  "date": "2025-06-25"
}
```

## 部署指南

1. 构建项目：`npm run build`
2. 将`dist`目录部署到Web服务器或AWS S3
3. 配置API代理，确保前端能够正确访问后端API 