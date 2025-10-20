"""
配置Elasticsearch并索引商品数据
"""

from elasticsearch import Elasticsearch
from tqdm import tqdm
import time

def setup_elasticsearch(esci_data, es_host="localhost", es_port=9200):
    """
    设置ES并索引商品
    """
    print("\n🔍 设置Elasticsearch...")
    
    # 连接ES
    try:
        es = Elasticsearch([{'host': es_host, 'port': es_port}])
        print(f"✓ 已连接到ES: {es_host}:{es_port}")
    except Exception as e:
        print(f"❌ ES连接失败: {e}")
        print("请先启动Elasticsearch:")
        print("  docker run -d -p 9200:9200 -e 'discovery.type=single-node' elasticsearch:8.10.0")
        return None
    
    # 创建索引
    index_name = "amazon_products"
    
    if es.indices.exists(index=index_name):
        print(f"✓ 索引 '{index_name}' 已存在")
        return es
    
    # 索引配置
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "product_id": {"type": "keyword"},
                "title": {"type": "text"},
                "description": {"type": "text"},
                "query": {"type": "text"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=settings)
    print(f"✓ 创建索引 '{index_name}'")
    
    # 索引商品数据
    print("📥 索引商品数据...")
    
    # 去重
    unique_products = {}
    for item in esci_data['train']:
        pid = item['product_id']
        if pid not in unique_products:
            unique_products[pid] = {
                'product_id': pid,
                'title': item['product_title'],
                'description': item.get('product_description', ''),
                'query': item['query']  # 保存一个示例query
            }
    
    # 批量索引
    for product in tqdm(unique_products.values()):
        es.index(index=index_name, body=product)
    
    print(f"✓ 已索引 {len(unique_products)} 个商品")
    
    # 等待索引完成
    time.sleep(2)
    es.indices.refresh(index=index_name)
    
    return es

def search_products(es, query, index_name="amazon_products", top_k=20):
    """
    搜索商品
    """
    results = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "description", "query"]
                }
            },
            "size": top_k
        }
    )
    
    products = []
    for hit in results['hits']['hits']:
        products.append(hit['_source'])
    
    return products

if __name__ == "__main__":
    from data.download_esci import download_esci_dataset
    
    dataset = download_esci_dataset()
    es = setup_elasticsearch(dataset)
    
    # 测试搜索
    if es:
        products = search_products(es, "summer dress")
        print(f"\n测试搜索 'summer dress': {len(products)} 个结果")
        for p in products[:3]:
            print(f"  - {p['title']}")