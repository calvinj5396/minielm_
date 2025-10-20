"""
在线DPO训练（Direct Preference Optimization）
使用强化学习优化查询改写质量
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random
from tqdm import tqdm
import os
from utils.llm_api import LLMClient
from utils.elasticsearch_utils import ElasticsearchClient
from models.relevance_scorer import RelevanceScorer

class OnlineDPOTrainer:
    def __init__(self,
                 model_path: str,
                 config: dict,
                 use_llm_feedback: bool = True):
        """
        初始化在线DPO训练器
        
        Args:
            model_path: MiniELM模型路径（蒸馏后）
            config: 配置字典
            use_llm_feedback: 是否使用LLM模拟用户反馈
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_llm_feedback = use_llm_feedback
        
        print("\n🎮 初始化在线DPO训练...")
        
        # 加载MiniELM
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.train()
        
        # 加载辅助模块
        self.relevance_scorer = RelevanceScorer.load_pretrained(
            "./checkpoints/relevance_scorer"
        )
        self.es_client = ElasticsearchClient()
        
        if use_llm_feedback:
            self.llm_client = LLMClient()
            print("   ✓ LLM反馈已启用")
        else:
            self.llm_client = None
            print("   ⚠ LLM反馈已禁用（仅使用相关性+多样性）")
        
        # DPO参数
        self.beta = config['training']['dpo']['beta']
        
        print("✓ 初始化完成")
    
    def compute_reward(self, 
                       query: str,
                       rewritten_query: str,
                       user_profile: dict = None) -> float:
        """
        计算奖励分数
        
        组合三个指标：
        1. 相关性 (Relevance)
        2. 多样性 (Diversity)
        3. 用户反馈 (User Feedback) - 可选
        """
        # 1. 搜索商品
        original_products = self.es_client.search(query)
        rewritten_products = self.es_client.search(rewritten_query)
        
        if len(rewritten_products) == 0:
            return 0.0  # 没有检索到商品，奖励为0
        
        # 2. 相关性得分
        relevance_scores = self.relevance_scorer.batch_score(
            query,
            rewritten_products
        )
        relevance = sum(relevance_scores) / len(relevance_scores)
        
        # 3. 多样性得分
        original_ids = set([p['product_id'] for p in original_products])
        rewritten_ids = set([p['product_id'] for p in rewritten_products])
        new_products = len(rewritten_ids - original_ids)
        diversity = new_products / len(original_ids) if len(original_ids) > 0 else 0
        
        # 4. 用户反馈得分（可选）
        if self.use_llm_feedback and user_profile:
            feedback = self.llm_client.simulate_user_feedback(
                query,
                rewritten_products,
                user_profile
            )
            user_score = (
                0.4 * feedback['click_rate'] +
                0.3 * feedback['add_to_cart_rate'] +
                0.3 * feedback['purchase_rate']
            )
        else:
            user_score = 0
        
        # 5. 综合奖励（权重可调）
        if self.use_llm_feedback:
            reward = (
                0.4 * relevance +
                0.3 * diversity +
                0.3 * user_score
            )
        else:
            reward = (
                0.6 * relevance +
                0.4 * diversity
            )
        
        return reward
    
    def dpo_loss(self, 
                 query_ids,
                 preferred_ids,
                 rejected_ids):
        """
        DPO损失函数（论文公式2）
        
        L_DPO = -1/B Σ log σ(β log(π_θ(Q~+|Q) / π_θ(Q~-|Q)))
        """
        # 计算preferred的log概率
        preferred_outputs = self.model(
            input_ids=query_ids,
            labels=preferred_ids
        )
        preferred_logprob = -preferred_outputs.loss
        
        # 计算rejected的log概率
        rejected_outputs = self.model(
            input_ids=query_ids,
            labels=rejected_ids
        )
        rejected_logprob = -rejected_outputs.loss
        
        # DPO损失
        logits_diff = self.beta * (preferred_logprob - rejected_logprob)
        loss = -F.logsigmoid(logits_diff).mean()
        
        return loss
    
    def train(self,
              queries: list,
              user_profiles: list,
              iterations: int = 500,
              learning_rate: float = 1e-5,
              output_dir: str = "./checkpoints/minielm_dpo"):
        """
        在线DPO训练
        
        Args:
            queries: 训练查询列表
            user_profiles: 用户画像列表
            iterations: 训练迭代次数
            learning_rate: 学习率
            output_dir: 输出目录
        """
        print(f"\n🚀 开始在线DPO训练...")
        print(f"   迭代次数: {iterations}")
        print(f"   查询数量: {len(queries)}")
        print(f"   LLM评估频率: 每{self.config['training']['dpo']['llm_eval_frequency']}步")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 训练循环
        pbar = tqdm(range(iterations), desc="DPO Training")
        total_loss = 0
        
        for iteration in pbar:
            # 1. 随机采样query和user profile
            query = random.choice(queries)
            user_profile = random.choice(user_profiles)
            
            # 2. 生成两个候选改写
            rewrite_1 = self.generate_rewrite(query, temperature=0.7)
            rewrite_2 = self.generate_rewrite(query, temperature=0.9)
            
            # 3. 计算奖励（每N步才用LLM）
            use_llm = (iteration % self.config['training']['dpo']['llm_eval_frequency'] == 0)
            
            if use_llm and self.use_llm_feedback:
                reward_1 = self.compute_reward(query, rewrite_1, user_profile)
                reward_2 = self.compute_reward(query, rewrite_2, user_profile)
            else:
                # 只用相关性+多样性（快速）
                reward_1 = self.compute_reward(query, rewrite_1, user_profile=None)
                reward_2 = self.compute_reward(query, rewrite_2, user_profile=None)
            
            # 4. 确定preferred和rejected
            if reward_1 > reward_2:
                preferred, rejected = rewrite_1, rewrite_2
            else:
                preferred, rejected = rewrite_2, rewrite_1
            
            # 5. Tokenize
            query_input = f"rewrite query: {query}"
            query_ids = self.tokenizer(
                query_input,
                return_tensors='pt',
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            preferred_ids = self.tokenizer(
                preferred,
                return_tensors='pt',
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            rejected_ids = self.tokenizer(
                rejected,
                return_tensors='pt',
                max_length=128,
                truncation=True
            ).input_ids.to(self.device)
            
            # 6. 计算DPO损失并更新
            loss = self.dpo_loss(query_ids, preferred_ids, rejected_ids)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 7. 记录
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(iteration+1):.4f}"
            })
            
            # 8. 定期保存
            if (iteration + 1) % 50 == 0:
                checkpoint_dir = f"{output_dir}/checkpoint-{iteration+1}"
                self.model.save_pretrained(checkpoint_dir)
                print(f"\n   💾 Checkpoint保存: {checkpoint_dir}")
        
        # 最终保存
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ DPO训练完成！模型已保存到: {output_dir}")
        
        return self.model
    
    def generate_rewrite(self, query: str, temperature: float = 0.8) -> str:
        """生成单个改写"""
        input_text = f"rewrite query: {query}"
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                do_sample=True,
                temperature=temperature,
                top_p=0.9
            )
        
        rewrite = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return rewrite


# ============ 用户画像生成 ============
def generate_user_profiles(num_profiles: int = 100) -> list:
    """
    生成模拟用户画像池
    """
    profiles = []
    
    genders = ['Male', 'Female', 'Other']
    ages = ['18-25', '26-35', '36-50', '50+']
    locations = ['North America', 'Europe', 'Asia']
    incomes = ['Low', 'Middle', 'High', 'Luxury']
    price_sensitivities = ['Low', 'Medium', 'High']
    brand_affinities = ['Low', 'Medium', 'High']
    styles = ['Casual', 'Business', 'Luxury', 'Trendy', 'Minimalist', 'Classic']
    
    for _ in range(num_profiles):
        profile = {
            'gender': random.choice(genders),
            'age': random.choice(ages),
            'location': random.choice(locations),
            'income': random.choice(incomes),
            'price_sensitivity': random.choice(price_sensitivities),
            'brand_affinity': random.choice(brand_affinities),
            'style': random.choice(styles)
        }
        profiles.append(profile)
    
    return profiles


# ============ 测试代码 ============
if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # 生成用户画像
    user_profiles = generate_user_profiles(100)
    
    # 准备训练查询（从ESCI中采样）
    test_queries = [
        "summer dress",
        "men shoes",
        "wireless headphones",
        "laptop bag",
        "coffee maker"
    ]
    
    # 初始化DPO训练器
    trainer = OnlineDPOTrainer(
        model_path="./checkpoints/minielm",
        config=config,
        use_llm_feedback=True
    )
    
    # 训练
    trainer.train(
        queries=test_queries,
        user_profiles=user_profiles,
        iterations=500
    )