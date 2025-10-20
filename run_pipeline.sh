#!/bin/bash

echo "========================================="
echo "  MiniELM 完整训练流程"
echo "========================================="

# 1. 下载数据
echo -e "\n📥 步骤1: 下载ESCI数据集..."
python data/download_esci.py

# 2. 构建Q2Q数据集
echo -e "\n🔨 步骤2: 构建Q2Q数据集..."
python data/build_q2q.py

# 3. 启动Elasticsearch (如果未启动)
echo -e "\n🔍 步骤3: 检查Elasticsearch..."
if ! curl -s http://localhost:9200 > /dev/null; then
    echo "启动Elasticsearch..."
    docker run -d -p 9200:9200 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    sleep 30
fi

# 4. 索引商品数据
echo -e "\n📦 步骤4: 索引商品数据..."
python data/setup_elasticsearch.py

# 5. 训练相关性模型
echo -e "\n🎯 步骤5: 训练相关性评分模型..."
python models/relevance_scorer.py

# 6. SFT训练Student
echo -e "\n🎒 步骤6: SFT训练Student模型..."
python training/sft_trainer.py

# 7. 知识蒸馏
echo -e "\n🔬 步骤7: 知识蒸馏..."
python -c "
from training.distillation import DistillationTrainer
from training.sft_trainer import SFTTrainer
import yaml

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 加载数据
trainer = SFTTrainer(config['models']['student']['name'])
train_data, val_data = trainer.load_q2q_data()

# 蒸馏
distiller = DistillationTrainer(
    teacher_path='./checkpoints/teacher_sft',
    student_path='./checkpoints/student_sft'
)
# TODO: 需要准备dataloader
print('请运行完整的Python脚本进行蒸馏')
"

# 8. 在线DPO训练
echo -e "\n🎮 步骤8: 在线DPO训练..."
python training/online_dpo.py

# 9. 评估
echo -e "\n📊 步骤9: 评估模型..."
python evaluation/evaluator.py

echo -e "\n✅ 完整流程执行完毕！"
echo "查看结果: ./outputs/"