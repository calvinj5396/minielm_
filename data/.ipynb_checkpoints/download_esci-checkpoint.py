"""
下载和预处理Amazon ESCI数据集
"""

from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm

def download_esci_dataset(output_dir: str = "./data/esci"):
    """
    下载ESCI数据集
    """
    print("📥 开始下载Amazon ESCI数据集...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载数据
    dataset = load_dataset("amazon_esci", "us")
    
    # 保存为CSV方便查看
    for split in ['train', 'test']:
        df = pd.DataFrame(dataset[split])
        df.to_csv(f"{output_dir}/{split}.csv", index=False)
        print(f"✓ {split} split: {len(df)} 条数据")
    
    print(f"\n✓ 数据已保存到: {output_dir}")
    return dataset

def preprocess_esci(dataset):
    """
    预处理ESCI数据
    - 只保留E(Exact)和S(Substitute)标签的数据
    - 构建query-product映射
    """
    print("\n🔧 预处理ESCI数据...")
    
    train_data = dataset['train']
    
    # 过滤相关商品
    relevant_data = []
    for item in tqdm(train_data):
        if item['esci_label'] in ['E', 'S']:
            relevant_data.append({
                'query': item['query'],
                'product_id': item['product_id'],
                'product_title': item['product_title'],
                'label': item['esci_label']
            })
    
    print(f"✓ 过滤后保留 {len(relevant_data)} 条相关数据")
    return relevant_data

if __name__ == "__main__":
    dataset = download_esci_dataset()
    relevant_data = preprocess_esci(dataset)