"""
构建Query-to-Query (Q2Q)数据集
"""

from collections import defaultdict
from tqdm import tqdm
import json
import random
from utils.llm_api import LLMClient

def build_q2q_dataset(relevant_data, 
                      min_common_products: int = 5,
                      max_pairs: int = 5000,
                      use_llm_filter: bool = True):
    """
    构建Q2Q数据集
    
    Args:
        relevant_data: 预处理后的ESCI数据
        min_common_products: 最少共同商品数
        max_pairs: 最多保留的query对数量
        use_llm_filter: 是否用LLM过滤
    """
    print("\n🔨 开始构建Q2Q数据集...")
    
    # Step 1: 构建query到products的映射
    print("Step 1: 构建Query-Product映射...")
    query_to_products = defaultdict(set)
    
    for item in tqdm(relevant_data):
        query_to_products[item['query']].add(item['product_id'])
    
    print(f"✓ 共有 {len(query_to_products)} 个不同的query")
    
    # Step 2: 找出有足够共同商品的query对
    print(f"\nStep 2: 寻找共享至少{min_common_products}个商品的query对...")
    queries = list(query_to_products.keys())
    candidate_pairs = []
    
    for i in tqdm(range(len(queries))):
        for j in range(i+1, len(queries)):
            q1, q2 = queries[i], queries[j]
            common = query_to_products[q1] & query_to_products[q2]
            
            if len(common) >= min_common_products:
                candidate_pairs.append({
                    'query1': q1,
                    'query2': q2,
                    'common_products': len(common)
                })
                
                if len(candidate_pairs) >= max_pairs * 2:
                    break
        if len(candidate_pairs) >= max_pairs * 2:
            break
    
    print(f"✓ 找到 {len(candidate_pairs)} 个候选query对")
    
    # Step 3: 用LLM过滤语义等价的query对（可选）
    if use_llm_filter and len(candidate_pairs) > 0:
        print(f"\nStep 3: 用LLM过滤语义等价的query对...")
        client = LLMClient()
        
        filtered_pairs = []
        for pair in tqdm(candidate_pairs[:max_pairs]):  # 限制LLM调用次数
            try:
                is_equiv = client.filter_query_pair(
                    pair['query1'], 
                    pair['query2']
                )
                if is_equiv:
                    filtered_pairs.append(pair)
            except Exception as e:
                print(f"⚠ 跳过: {e}")
                continue
        
        print(f"✓ LLM过滤后保留 {len(filtered_pairs)} 个query对")
    else:
        print("\nStep 3: 跳过LLM过滤（use_llm_filter=False）")
        filtered_pairs = candidate_pairs[:max_pairs]
    
    # Step 4: 构建训练数据（双向）
    q2q_data = []
    for pair in filtered_pairs:
        # query1 -> query2
        q2q_data.append({
            'query': pair['query1'],
            'rewritten_query': pair['query2']
        })
        # query2 -> query1 (双向)
        q2q_data.append({
            'query': pair['query2'],
            'rewritten_query': pair['query1']
        })
    
    print(f"\n✓ 最终Q2Q数据集: {len(q2q_data)} 条")
    
    # 保存
    output_path = "./data/q2q_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(q2q_data, f, indent=2)
    
    print(f"✓ 已保存到: {output_path}")
    return q2q_data

if __name__ == "__main__":
    # 加载预处理数据
    import sys
    sys.path.append('.')
    from data.download_esci import download_esci_dataset, preprocess_esci
    
    dataset = download_esci_dataset()
    relevant_data = preprocess_esci(dataset)
    
    # 构建Q2Q
    q2q_data = build_q2q_dataset(
        relevant_data,
        min_common_products=5,
        max_pairs=5000,
        use_llm_filter=True  # 改为False可以跳过LLM，更快但质量略低
    )