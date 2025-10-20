# MiniELM: Query Rewriting for E-Commerce

论文复现项目：轻量级、自适应的电商查询改写框架

## 🚀 快速开始

### 1. AutoDL环境配置

**选择机器**：RTX 4090 (24GB) - 推荐
- 价格：~2.5元/小时
- 总耗时：12-15小时
- 总成本：30-40元

**创建实例后执行**：
```bash
# 克隆项目
git clone https://github.com/yourusername/minielm.git
cd minielm

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 启动Elasticsearch (Docker)
docker run -d -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.10.0
```

### 2. 配置API Key

**获取OpenAI API Key**：
1. 访问 https://platform.openai.com/api-keys
2. 创建新的API Key
3. 复制key

**或使用中转服务**（推荐，更便宜）：
- https://www.openai-sb.com
- https://api.chatanywhere.tech

**填写到配置文件**：
编辑 `config/config.yaml`，找到这一行：
```yaml
api:
  openai:
    api_key: "sk-YOUR_OPENAI_KEY_HERE"  # 👈 粘贴你的key
```

### 3. 运行完整流程
```bash
# 方式1: 一键运行
bash run_pipeline.sh

# 方式2: 分步执行
python data/download_esci.py          # 下载数据
python data/build_q2q.py              # 构建Q2Q (会调用API)
python data/setup_elasticsearch.py     # 索引商品
python models/relevance_scorer.py     # 训练相关性模型
python training/sft_trainer.py        # SFT训练
# ...依次执行
```

## 📁 项目结构
```
minielm_project/
├── config/config.yaml          # 配置文件（API key在这里）
├── data/                       # 数据处理
├── models/                     # 模型定义
├── training/                   # 训练脚本
├── evaluation/                 # 评估脚本
├── utils/                      # 工具函数
│   ├── llm_api.py             # 👈 API调用核心
│   └── elasticsearch_utils.py
└── requirements.txt
```

## 💰 成本估算

| 项目 | 成本 |
|------|------|
| GPU租用 (RTX 4090, 15h) | ¥38 |
| OpenAI API调用 | <¥0.5 |
| **总计** | **¥40以内** |

## 📊 预期结果

- **商品覆盖率**: +30-35%
- **相关性**: +12-15%
- **多样性**: +15-18%

## ⚠️ 常见问题

**Q: API调用失败**
```python
# 检查config.yaml中的API key是否正确
# 检查网络连接
# 如果使用中转服务，确认base_url正确
```

**Q: Elasticsearch连接失败**
```bash
# 检查Docker是否启动
docker ps

# 如果没有启动ES
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.10.0
```

**Q: 显存不足**
```yaml
# 修改config.yaml
training:
  sft:
    batch_size: 4  # 降低batch size
    gradient_accumulation_steps: 8  # 增加梯度累积
```

## 📝 引用
```bibtex
@article{minielm2025,
  title={MiniELM: A Lightweight and Adaptive Query Rewriting Framework for E-Commerce Search Optimization},
  author={...},
  journal={arXiv preprint arXiv:2501.18056},
  year={2025}
}
```