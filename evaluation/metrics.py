"""
评估指标计算
"""

import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer

class Metrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def exact_match(self, predictions: list, references: list) -> float:
        """ExactMatch: 完全匹配率"""
        matches = sum([
            pred.strip().lower() == ref.strip().lower()
            for pred, ref in zip(predictions, references)
        ])
        return matches / len(predictions)
    
    def rouge_l(self, predictions: list, references: list) -> float:
        """RougeL: 最长公共子序列F1"""
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores.append(score['rougeL'].fmeasure)
        return np.mean(scores)
    
    def product_coverage(self, 
                        all_rewritten_products: list,
                        relevance_labels: dict) -> int:
        """
        商品覆盖率: 不重复的相关商品数
        
        Args:
            all_rewritten_products: 所有改写查询检索到的商品列表
            relevance_labels: {product_id: label}
        """
        relevant_products = set()
        
        for product in all_rewritten_products:
            pid = product['product_id']
            label = relevance_labels.get(pid, 'I')
            if label in ['E', 'S']:  # 只统计相关商品
                relevant_products.add(pid)
        
        return len(relevant_products)
    
    def relevance_score(self, 
                       query: str,
                       products: list,
                       relevance_scorer) -> float:
        """相关性得分"""
        if len(products) == 0:
            return 0.0
        
        scores = relevance_scorer.batch_score(query, products)
        return np.mean(scores)
    
    def diversity_score(self,
                       original_products: list,
                       rewritten_products: list) -> float:
        """多样性得分"""
        original_ids = set([p['product_id'] for p in original_products])
        rewritten_ids = set([p['product_id'] for p in rewritten_products])
        
        if len(original_ids) == 0:
            return 0.0
        
        new_products = len(rewritten_ids - original_ids)
        return new_products / len(original_ids)
    
    def user_feedback_scores(self,
                           query: str,
                           products: list,
                           user_profile: dict,
                           llm_client) -> dict:
        """用户反馈得分"""
        feedback = llm_client.simulate_user_feedback(
            query,
            products,
            user_profile
        )
        return feedback


# ============ 测试代码 ============
if __name__ == "__main__":
    metrics = Metrics()
    
    # 测试ExactMatch
    preds = ["summer dress for women", "men shoes"]
    refs = ["summer dress for women", "shoes for men"]
    em = metrics.exact_match(preds, refs)
    print(f"ExactMatch: {em:.3f}")
    
    # 测试RougeL
    rl = metrics.rouge_l(preds, refs)
    print(f"RougeL: {rl:.3f}")