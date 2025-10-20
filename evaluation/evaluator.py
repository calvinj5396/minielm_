"""
完整评估流程
"""

import json
from tqdm import tqdm
import random
from evaluation.metrics import Metrics
from models.relevance_scorer import RelevanceScorer
from utils.elasticsearch_utils import ElasticsearchClient
from utils.llm_api import LLMClient
from training.online_dpo import generate_user_profiles

class Evaluator:
    def __init__(self, 
                 model,
                 tokenizer,
                 config: dict,
                 use_llm_feedback: bool = True):
        """初始化评估器"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # 加载辅助模块
        self.metrics = Metrics()
        self.relevance_scorer = RelevanceScorer.load_pretrained(
            "./checkpoints/relevance_scorer"
        )
        self.es_client = ElasticsearchClient()
        
        if use_llm_feedback:
            self.llm_client = LLMClient()
        else:
            self.llm_client = None
        
        # 生成用户画像
        self.user_profiles = generate_user_profiles(50)
    
    def evaluate_offline(self, test_data: list) -> dict:
        """
        离线评估（基于Q2Q数据集）
        
        Metrics: ExactMatch, RougeL, CrossEntropy
        """
        print("\n📊 离线评估...")
        
        predictions = []
        references = []
        
        for item in tqdm(test_data[:100], desc="Generating"):
            query = item['query']
            reference = item['rewritten_query']
            
            # 生成改写
            input_text = f"rewrite query: {query}"
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
            
            outputs = self.model.generate(**inputs, max_length=128)
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            predictions.append(prediction)
            references.append(reference)
        
        # 计算指标
        results = {
            'exact_match': self.metrics.exact_match(predictions, references),
            'rouge_l': self.metrics.rouge_l(predictions, references)
        }
        
        print(f"✓ ExactMatch: {results['exact_match']:.4f}")
        print(f"✓ RougeL: {results['rouge_l']:.4f}")
        
        return results
    
    def evaluate_online(self, test_queries: list, num_rewrites: int = 10) -> dict:
        """
        在线评估
        
        Metrics: Product Coverage, Relevance, Diversity, User Feedback
        """
        print("\n📊 在线评估...")
        
        all_covered_products = []
        relevance_scores = []
        diversity_scores = []
        click_rates = []
        add_to_cart_rates = []
        purchase_rates = []
        
        for query in tqdm(test_queries, desc="Evaluating"):
            # 1. 生成多个改写
            input_text = f"rewrite query: {query}"
            inputs = self.tokenizer(input_text, return_tensors='pt').to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_return_sequences=num_rewrites,
                do_sample=True,
                temperature=0.8
            )
            
            rewrites = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
            # 2. 对每个改写计算指标
            original_products = self.es_client.search(query)
            
            for rewrite in rewrites:
                rewritten_products = self.es_client.search(rewrite)
                
                if len(rewritten_products) == 0:
                    continue
                
                # 商品覆盖
                all_covered_products.extend(rewritten_products)
                
                # 相关性
                relevance = self.metrics.relevance_score(
                    query,
                    rewritten_products,
                    self.relevance_scorer
                )
                relevance_scores.append(relevance)
                
                # 多样性
                diversity = self.metrics.diversity_score(
                    original_products,
                    rewritten_products
                )
                diversity_scores.append(diversity)
                
                # 用户反馈（可选）
                if self.llm_client:
                    user_profile = random.choice(self.user_profiles)
                    feedback = self.metrics.user_feedback_scores(
                        query,
                        rewritten_products,
                        user_profile,
                        self.llm_client
                    )
                    click_rates.append(feedback['click_rate'])
                    add_to_cart_rates.append(feedback['add_to_cart_rate'])
                    purchase_rates.append(feedback['purchase_rate'])
        
        # 汇总结果
        results = {
            'product_coverage': len(set([p['product_id'] for p in all_covered_products])),
            'relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'diversity': sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        }
        
        if self.llm_client:
            results.update({
                'click_rate': sum(click_rates) / len(click_rates) if click_rates else 0,
                'add_to_cart_rate': sum(add_to_cart_rates) / len(add_to_cart_rates) if add_to_cart_rates else 0,
                'purchase_rate': sum(purchase_rates) / len(purchase_rates) if purchase_rates else 0
            })
        
        # 打印结果
        print(f"✓ Product Coverage: {results['product_coverage']}")
        print(f"✓ Relevance: {results['relevance']:.4f}")
        print(f"✓ Diversity: {results['diversity']:.4f}")
        
        if self.llm_client:
            print(f"✓ Click Rate: {results['click_rate']:.4f}")
            print(f"✓ Add-to-Cart Rate: {results['add_to_cart_rate']:.4f}")
            print(f"✓ Purchase Rate: {results['purchase_rate']:.4f}")
        
        return results
    
    def save_results(self, results: dict, output_path: str):
        """保存评估结果"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 结果已保存到: {output_path}")


# ============ 测试代码 ============
if __name__ == "__main__":
    print("请使用run_pipeline.py运行完整评估")