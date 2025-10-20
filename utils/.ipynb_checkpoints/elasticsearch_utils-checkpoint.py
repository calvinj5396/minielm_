"""
Elasticsearch工具函数
"""

from elasticsearch import Elasticsearch
import time

class ElasticsearchClient:
    def __init__(self, host="localhost", port=9200):
        """初始化ES客户端"""
        try:
            self.es = Elasticsearch([{'host': host, 'port': port, 'scheme': 'http'}])
            self.index_name = "amazon_products"
            
            # 测试连接
            if not self.es.ping():
                raise ConnectionError("无法连接到Elasticsearch")
            
            print(f"✓ ES连接成功: {host}:{port}")
        
        except Exception as e:
            print(f"❌ ES连接失败: {e}")
            print("请先启动Elasticsearch:")
            print("  docker run -d -p 9200:9200 -e 'discovery.type=single-node' docker.elastic.co/elasticsearch/elasticsearch:8.10.0")
            raise
    
    def search(self, query: str, top_k: int = 20) -> list:
        """
        搜索商品
        
        Returns:
            List of products: [{'product_id': ..., 'title': ..., ...}]
        """
        try:
            results = self.es.search(
                index=self.index_name,
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
        
        except Exception as e:
            print(f"⚠ 搜索失败: {e}")
            return []
    
    def index_exists(self) -> bool:
        """检查索引是否存在"""
        return self.es.indices.exists(index=self.index_name)


# ============ 测试代码 ============
if __name__ == "__main__":
    client = ElasticsearchClient()
    
    # 测试搜索
    products = client.search("summer dress", top_k=5)
    print(f"\n搜索结果: {len(products)} 个商品")
    for p in products[:3]:
        print(f"  - {p['title']}")