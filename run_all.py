"""
完整的端到端运行脚本
按顺序执行所有步骤
"""

import os
import sys
import yaml
import json
from datetime import datetime

def print_step(step_num, title):
    """打印步骤标题"""
    print("\n" + "="*60)
    print(f"步骤 {step_num}: {title}")
    print("="*60 + "\n")

def main():
    print("🚀 MiniELM 完整训练流程开始")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # ========== 步骤1: 下载数据 ==========
    print_step(1, "下载ESCI数据集")
    from data.download_esci import download_esci_dataset, preprocess_esci
    
    dataset = download_esci_dataset()
    relevant_data = preprocess_esci(dataset)
    
    # ========== 步骤2: 构建Q2Q ==========
    print_step(2, "构建Q2Q数据集")
    from data.build_q2q import build_q2q_dataset
    
    q2q_data = build_q2q_dataset(
        relevant_data,
        min_common_products=config['data']['q2q_min_common_products'],
        max_pairs=5000,
        use_llm_filter=True  # 会调用API
    )
    
    # ========== 步骤3: 设置ES ==========
    print_step(3, "设置Elasticsearch")
    from data.setup_elasticsearch import setup_elasticsearch
    
    es = setup_elasticsearch(dataset)
    
    if es is None:
        print("❌ Elasticsearch未启动，请先启动ES")
        sys.exit(1)
    
    # ========== 步骤4: 训练相关性模型 ==========
    print_step(4, "训练相关性评分模型")
    from models.relevance_scorer import RelevanceScorer
    
    relevance_scorer = RelevanceScorer(
        model_name=config['models']['relevance_scorer']['name']
    )
    train_data, val_data = relevance_scorer.prepare_dataset(dataset)
    relevance_scorer.train(train_data, val_data)
    
    # ========== 步骤5: SFT训练Student ==========
    print_step(5, "SFT训练Student模型")
    from training.sft_trainer import SFTTrainer
    
    student_trainer = SFTTrainer(
        model_name=config['models']['student']['name'],
        is_teacher=False
    )
    train_q2q, val_q2q = student_trainer.load_q2q_data()
    student_trainer.train(
        train_q2q,
        val_q2q,
        num_epochs=config['training']['sft']['num_epochs'],
        batch_size=config['training']['sft']['batch_size'],
        learning_rate=config['training']['sft']['learning_rate']
    )
    
    # ========== 步骤6: 知识蒸馏 ==========
    print_step(6, "知识蒸馏（Teacher → Student）")
    from training.distillation import DistillationTrainer
    from torch.utils.data import DataLoader
    
    # 注意：这里需要准备好dataloader
    # 简化版：直接加载已有的Student作为MiniELM
    print("⚠ 跳过蒸馏步骤（可选优化）")
    print("如需完整蒸馏，请手动运行training/distillation.py")
    
    # ========== 步骤7: 在线DPO ==========
    print_step(7, "在线DPO训练")
    from training.online_dpo import OnlineDPOTrainer, generate_user_profiles
    
    # 准备训练数据
    train_queries = [item['query'] for item in q2q_data[:1000]]  # 取前1000个query
    user_profiles = generate_user_profiles(100)
    
    dpo_trainer = OnlineDPOTrainer(
        model_path="./checkpoints/student_sft",  # 使用SFT后的模型
        config=config,
        use_llm_feedback=True  # 会调用API
    )
    
    dpo_trainer.train(
        queries=train_queries,
        user_profiles=user_profiles,
        iterations=config['training']['dpo']['iterations'],
        learning_rate=config['training']['dpo']['learning_rate']
    )
    
    # ========== 步骤8: 评估 ==========
    print_step(8, "模型评估")
    from evaluation.evaluator import Evaluator
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    # 加载最终模型
    final_model = AutoModelForSeq2SeqLM.from_pretrained("./checkpoints/minielm_dpo")
    final_tokenizer = AutoTokenizer.from_pretrained("./checkpoints/minielm_dpo")
    
    evaluator = Evaluator(
        model=final_model,
        tokenizer=final_tokenizer,
        config=config,
        use_llm_feedback=True
    )
    
    # 离线评估
    test_q2q = val_q2q.select(range(100))  # 取100个样本
    offline_results = evaluator.evaluate_offline(test_q2q)
    
    # 在线评估
    test_queries = list(set([item['query'] for item in q2q_data[-500:]]))[:100]
    online_results = evaluator.evaluate_online(test_queries, num_rewrites=10)
    
    # 保存结果
    all_results = {
        'offline': offline_results,
        'online': online_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = config['system']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator.save_results(all_results, f"{output_dir}/final_results.json")
    
    # ========== 完成 ==========
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n结果摘要:")
    print(f"  离线 - ExactMatch: {offline_results['exact_match']:.4f}")
    print(f"  离线 - RougeL: {offline_results['rouge_l']:.4f}")
    print(f"  在线 - Product Coverage: {online_results['product_coverage']}")
    print(f"  在线 - Relevance: {online_results['relevance']:.4f}")
    print(f"  在线 - Diversity: {online_results['diversity']:.4f}")
    
    if 'click_rate' in online_results:
        print(f"  在线 - Click Rate: {online_results['click_rate']:.4f}")
        print(f"  在线 - Add-to-Cart Rate: {online_results['add_to_cart_rate']:.4f}")
        print(f"  在线 - Purchase Rate: {online_results['purchase_rate']:.4f}")
    
    print(f"\n📁 详细结果已保存到: {output_dir}/final_results.json")
    print(f"📁 模型已保存到: ./checkpoints/minielm_dpo")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()