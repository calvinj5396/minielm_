"""
监督微调（SFT）训练器
用于在Q2Q数据集上训练Teacher和Student
"""

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json

class SFTTrainer:
    def __init__(self, model_name: str, is_teacher: bool = False):
        """
        初始化SFT训练器
        
        Args:
            model_name: 模型名称（如google/flan-t5-base）
            is_teacher: 是否是Teacher模型
        """
        self.model_name = model_name
        self.is_teacher = is_teacher
        
        print(f"\n{'🎓 Teacher' if is_teacher else '🎒 Student'} 模型初始化: {model_name}")
        
        # 加载模型和tokenizer
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 统计参数量
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"   参数量: {num_params/1e6:.1f}M")
    
    def load_q2q_data(self, data_path: str = "./data/q2q_dataset.json"):
        """加载Q2Q数据集"""
        with open(data_path, 'r') as f:
            q2q_data = json.load(f)
        
        print(f"✓ 加载Q2Q数据: {len(q2q_data)} 条")
        
        # 转为Dataset
        dataset = Dataset.from_list(q2q_data)
        
        # 划分train/val
        split = dataset.train_test_split(test_size=0.1, seed=42)
        
        return split['train'], split['test']
    
    def preprocess_function(self, examples):
        """预处理函数：将query转为input_ids，rewritten_query转为labels"""
        
        # 输入：原始query
        # 添加任务前缀（T5风格）
        inputs = [f"rewrite query: {q}" for q in examples['query']]
        
        # 目标：改写后的query
        targets = examples['rewritten_query']
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs
    
    def train(self, 
              train_dataset,
              val_dataset,
              output_dir: str = None,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 5e-5):
        """
        训练模型
        """
        if output_dir is None:
            output_dir = f"./checkpoints/{'teacher' if self.is_teacher else 'student'}_sft"
        
        print(f"\n🚀 开始训练...")
        print(f"   输出目录: {output_dir}")
        
        # 预处理数据
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            gradient_accumulation_steps=4,  # 有效batch_size = 8*4=32
            learning_rate=learning_rate,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_total_limit=2,
            fp16=True,  # 混合精度
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最终模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✓ 训练完成！模型已保存到: {output_dir}")
        
        return trainer
    
    def generate(self, query: str, num_return_sequences: int = 1, **kwargs) -> list:
        """
        生成改写query
        """
        input_text = f"rewrite query: {query}"
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            **kwargs
        )
        
        rewrites = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return rewrites


# ============ 测试代码 ============
if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 训练Student模型
    trainer = SFTTrainer(
        model_name=config['models']['student']['name'],
        is_teacher=False
    )
    
    # 加载数据
    train_data, val_data = trainer.load_q2q_data()
    
    # 训练
    trainer.train(train_data, val_data)
    
    # 测试生成
    test_query = "summer dress"
    rewrites = trainer.generate(test_query, num_return_sequences=3)
    print(f"\n测试生成:\n原始: {test_query}")
    for i, r in enumerate(rewrites):
        print(f"改写{i+1}: {r}")