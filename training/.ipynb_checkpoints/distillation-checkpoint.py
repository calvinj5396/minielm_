"""
知识蒸馏训练器
使用反向KL散度将Teacher知识迁移到Student
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class DistillationTrainer:
    def __init__(self, 
                 teacher_path: str,
                 student_path: str,
                 temperature: float = 2.0):
        """
        初始化蒸馏训练器
        
        Args:
            teacher_path: Teacher模型路径
            student_path: Student模型路径（已SFT训练）
            temperature: 蒸馏温度
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        
        print("\n🔬 初始化知识蒸馏...")
        
        # 加载Teacher（eval模式，不训练）
        print(f"   加载Teacher: {teacher_path}")
        self.teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to(self.device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # 加载Student（训练模式）
        print(f"   加载Student: {student_path}")
        self.student = AutoModelForSeq2SeqLM.from_pretrained(student_path).to(self.device)
        self.student.train()
        
        # Tokenizer（共用）
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_path)
        
        print("✓ 模型加载完成")
    
    def reverse_kl_loss(self, student_logits, teacher_logits):
        """
        反向KL散度损失（论文公式1）
        
        D_KL(P_S || P_T) = Σ P_S(x) log(P_S(x) / P_T(x))
        
        让Student专注于Teacher的高概率区域
        """
        # 应用温度softmax
        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 反向KL散度
        kl_loss = F.kl_div(
            student_probs.log(),
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )
        
        # 温度校正
        return kl_loss * (self.temperature ** 2)
    
    def train(self,
              train_dataloader,
              val_dataloader,
              num_epochs: int = 3,
              learning_rate: float = 3e-5,
              output_dir: str = "./checkpoints/minielm"):
        """
        执行蒸馏训练
        """
        print(f"\n🎯 开始知识蒸馏训练...")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Temperature: {self.temperature}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate
        )
        
        # 混合精度
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n📖 Epoch {epoch+1}/{num_epochs}")
            
            # ========== 训练阶段 ==========
            self.student.train()
            train_loss = 0
            
            pbar = tqdm(train_dataloader, desc="Training")
            for batch_idx, batch in enumerate(pbar):
                # 将batch移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast():
                    # Teacher生成soft labels（不计算梯度）
                    with torch.no_grad():
                        teacher_outputs = self.teacher(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        teacher_logits = teacher_outputs.logits
                    
                    # Student前向传播
                    student_outputs = self.student(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    student_logits = student_outputs.logits
                    
                    # 计算反向KL损失
                    loss = self.reverse_kl_loss(student_logits, teacher_logits)
                
                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                # 定期保存checkpoint
                if (batch_idx + 1) % 500 == 0:
                    self.save_checkpoint(output_dir, epoch, batch_idx)
            
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"   训练损失: {avg_train_loss:.4f}")
            
            # ========== 验证阶段 ==========
            self.student.eval()
            val_loss = 0
            
            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc="Validation")
                for batch in pbar:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Teacher logits
                    teacher_outputs = self.teacher(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    
                    # Student logits
                    student_outputs = self.student(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    
                    loss = self.reverse_kl_loss(
                        student_outputs.logits,
                        teacher_outputs.logits
                    )
                    
                    val_loss += loss.item()
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"   验证损失: {avg_val_loss:.4f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"   💾 保存最佳模型...")
                self.student.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ 蒸馏完成！MiniELM已保存到: {output_dir}")
        return self.student
    
    def save_checkpoint(self, output_dir, epoch, step):
        """保存checkpoint"""
        checkpoint_dir = f"{output_dir}/checkpoint-epoch{epoch}-step{step}"
        self.student.save_pretrained(checkpoint_dir)
        print(f"   💾 Checkpoint保存: {checkpoint_dir}")


# ============ 测试代码 ============
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from training.sft_trainer import SFTTrainer
    import json
    
    # 1. 加载Q2Q数据
    with open("./data/q2q_dataset.json", 'r') as f:
        q2q_data = json.load(f)
    
    # 2. 准备dataloader（简化版，实际应该用上面的preprocess）
    # 这里假设你已经有了tokenized的数据
    
    # 3. 初始化蒸馏器
    distiller = DistillationTrainer(
        teacher_path="./checkpoints/teacher_sft",
        student_path="./checkpoints/student_sft",
        temperature=2.0
    )
    
    # 4. 训练（需要准备好dataloader）
    # distiller.train(train_dataloader, val_dataloader)
    
    print("请参考完整的run_pipeline.py脚本运行蒸馏")