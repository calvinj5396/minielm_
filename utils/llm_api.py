"""
LLM API调用模块
支持OpenAI和Anthropic，自动失败重试
"""

import os
import json
import time
from typing import Dict, List, Optional
import yaml
import openai
from anthropic import Anthropic

class LLMClient:
    def __init__(self, config_path: str = "config/config.yaml"):
        """初始化LLM客户端"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['api']['provider']
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openai(self):
        """初始化OpenAI客户端"""
        api_config = self.config['api']['openai']
        
        # 设置API Key
        openai.api_key = api_config['api_key']
        
        # 如果使用中转服务（如openai-sb.com），修改base_url
        if api_config.get('base_url'):
            openai.api_base = api_config['base_url']
        
        self.model = api_config['model']
        print(f"✓ OpenAI初始化成功，使用模型: {self.model}")
    
    def _init_anthropic(self):
        """初始化Anthropic客户端"""
        api_config = self.config['api']['anthropic']
        self.client = Anthropic(api_key=api_config['api_key'])
        self.model = api_config['model']
        print(f"✓ Anthropic初始化成功，使用模型: {self.model}")
    
    def call(self, 
             prompt: str, 
             max_tokens: int = 100,
             temperature: float = 0.7,
             max_retries: int = 3) -> str:
        """
        调用LLM API（带重试机制）
        
        Args:
            prompt: 提示词
            max_tokens: 最大生成token数
            temperature: 温度参数
            max_retries: 最大重试次数
        
        Returns:
            LLM生成的文本
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(prompt, max_tokens, temperature)
                else:
                    return self._call_anthropic(prompt, max_tokens, temperature)
            
            except Exception as e:
                print(f"⚠ API调用失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise
    
    def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """调用OpenAI API"""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for e-commerce query analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    def _call_anthropic(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """调用Anthropic API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()
    
    def simulate_user_feedback(self, 
                              query: str, 
                              products: List[Dict],
                              user_profile: Dict) -> Dict[str, float]:
        """
        模拟用户反馈（论文中的M2模型）
        
        Returns:
            {'click_rate': 0.5, 'add_to_cart_rate': 0.2, 'purchase_rate': 0.1}
        """
        # 只取前10个商品，减少token消耗
        product_titles = [p['title'] for p in products[:10]]
        
        prompt = f"""You are simulating an e-commerce user with this profile:
{json.dumps(user_profile, indent=2)}

The user searches for: "{query}"

The search returns these products:
{json.dumps(product_titles, indent=2)}

Based on the user profile and query, estimate how many products (0-10) the user would:
1. Click on
2. Add to cart
3. Purchase

Respond ONLY with valid JSON in this exact format:
{{"click": <number>, "add_to_cart": <number>, "purchase": <number>}}"""

        try:
            response = self.call(prompt, max_tokens=50, temperature=0.7)
            
            # 解析JSON
            feedback = json.loads(response)
            
            # 归一化为比例
            num_products = len(products)
            return {
                'click_rate': feedback['click'] / num_products,
                'add_to_cart_rate': feedback['add_to_cart'] / num_products,
                'purchase_rate': feedback['purchase'] / num_products
            }
        
        except json.JSONDecodeError:
            print(f"⚠ JSON解析失败，使用默认值: {response}")
            return {
                'click_rate': 0.3,
                'add_to_cart_rate': 0.1,
                'purchase_rate': 0.05
            }
    
    def filter_query_pair(self, query1: str, query2: str) -> bool:
        """
        判断两个query是否语义等价（用于构建Q2Q数据集）
        
        Returns:
            True if semantically equivalent
        """
        prompt = f"""Are these two search queries semantically equivalent?

Query 1: "{query1}"
Query 2: "{query2}"

Consider them equivalent if they express the same search intent, even with different words.

Respond with ONLY "Yes" or "No"."""

        try:
            response = self.call(prompt, max_tokens=10, temperature=0.3)
            return "yes" in response.lower()
        except:
            return False


# ============ 测试代码 ============
if __name__ == "__main__":
    # 测试API连接
    client = LLMClient()
    
    # 测试1: 基础调用
    response = client.call("Say 'Hello, API works!'")
    print(f"\n测试1 - 基础调用:\n{response}\n")
    
    # 测试2: 用户反馈模拟
    test_products = [
        {'title': 'Summer Maxi Dress for Women'},
        {'title': 'Casual Floral Sundress'},
        {'title': 'Beach Cover Up Dress'}
    ]
    test_profile = {
        'gender': 'Female',
        'age': '26-35',
        'style': 'Casual',
        'price_sensitivity': 'Medium'
    }
    
    feedback = client.simulate_user_feedback(
        query="summer dress",
        products=test_products,
        user_profile=test_profile
    )
    print(f"测试2 - 用户反馈:\n{json.dumps(feedback, indent=2)}\n")
    
    # 测试3: Query等价判断
    is_equiv = client.filter_query_pair("men shoes", "shoes for men")
    print(f"测试3 - Query等价: {is_equiv}\n")