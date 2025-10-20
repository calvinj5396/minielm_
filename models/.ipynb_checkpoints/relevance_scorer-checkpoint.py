"""
相关性评分模型（论文中的M1）
基于BERT微调，用于评估query-product相关性
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

class RelevanceScorer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """初始化相关性评分器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载BERT模型（4分类：E, S, C, I）
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4
        ).to(self.device)
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 标签映射
        self.label_to_id = {'E': 0, 'S': 1, 'C': 2, 'I': 3}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        # 分数映射（用于计算相关性）
        self.label_to_score = {'E': 1.0, 'S': 0.8, 'C': 0.3, 'I': 0.0}
    
    def prepare_dataset(self, esci_data):
        """
        准备训练数据
        从ESCI数据集构建(query, product_title, label)三元组
        """
        train_examples = []
        
        for item in esci_data['train']:
            train_examples.append({
                'query': item['query'],
                'product_title': item['product_title'],
                'label': self.label_to_id[item['esci_label']]
            })
        
        # 转为Hugging Face Dataset
        dataset = Dataset.from_list(train_examples)
        
        def tokenize_function(examples):
            # 输入格式: [CLS] query [SEP] product_title [SEP]
            return self.tokenizer(
                examples['query'],
                examples['product_title'],
                truncation=True,
                padding='max_length',
                max_length=128
            )
        
        tokenized = dataset.map(tokenize_function, batched=True)
        tokenized = tokenized.rename_column("label", "labels")
        
        # 划分train/val
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        
        return split['train'], split['test']
    
    def train(self, train_dataset, val_dataset, output_dir="./checkpoints/relevance_scorer"):
        """训练相关性评分器"""
        print("\n🎯 训练相关性评分模型...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            learning_rate=2e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            fp16=True,  # 混合精度加速
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )
        
        trainer.train()
        
        # 保存最终模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ 模型已保存到: {output_dir}")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        accuracy = (predictions == labels).mean()
        
        return {"accuracy": accuracy}
    
    def score(self, query: str, product_title: str) -> float:
        """
        给query-product对打分
        
        Returns:
            0-1之间的相关性分数
        """
        self.model.eval()
        
        inputs = self.tokenizer(
            query,
            product_title,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_label_id = torch.argmax(probs, dim=-1).item()
        
        # 转为标签
        predicted_label = self.id_to_label[predicted_label_id]
        
        # 返回分数
        return self.label_to_score[predicted_label]
    
    def batch_score(self, query: str, products: list) -> list:
        """批量打分（更高效）"""
        self.model.eval()
        
        scores = []
        
        # 批量处理
        batch_size = 32
        for i in range(0, len(products), batch_size):
            batch_products = products[i:i+batch_size]
            
            inputs = self.tokenizer(
                [query] * len(batch_products),
                [p['title'] for p in batch_products],
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(probs, dim=-1)
            
            # 转为分数
            for label_id in predicted_labels:
                label = self.id_to_label[label_id.item()]
                scores.append(self.label_to_score[label])
        
        return scores
    
    @classmethod
    def load_pretrained(cls, model_path: str):
        """加载已训练的模型"""
        scorer = cls()
        scorer.model = BertForSequenceClassification.from_pretrained(model_path).to(scorer.device)
        scorer.tokenizer = BertTokenizer.from_pretrained(model_path)
        print(f"✓ 加载模型: {model_path}")
        return scorer


# ============ 测试代码 ============
if __name__ == "__main__":
    from data.download_esci import download_esci_dataset
    
    # 下载数据
    dataset = download_esci_dataset()
    
    # 初始化scorer
    scorer = RelevanceScorer()
    
    # 准备数据
    train_data, val_data = scorer.prepare_dataset(dataset)
    
    # 训练
    scorer.train(train_data, val_data)
    
    # 测试
    test_query = "summer dress"
    test_product = "Women's Floral Maxi Summer Dress"
    score = scorer.score(test_query, test_product)
    print(f"\n测试相关性:\nQuery: {test_query}\nProduct: {test_product}\nScore: {score:.3f}")