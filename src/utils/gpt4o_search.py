"""
简单的 GPT-4o 客户端

这个脚本提供了一个简单的类，用于与 OpenAI 的 GPT-4o 模型通信，
主要用于解析搜索项和进行简单对话。
"""

import os
import re
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

# 加载环境变量（如果有.env文件）
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
class GPT4oClient:
    """
    与 OpenAI GPT-4o 模型交互的简单客户端。
    
    这个类提供方法来发送查询并解析响应，特别适用于搜索项解析。
    """
    
    def __init__(self, api_key: Optional[str] = None,base_url:str = openai_base_url, model: str = "gpt-4o-mini"):
        """
        初始化 GPT-4o 客户端
        
        参数:
            api_key: OpenAI API 密钥（如果为 None，将从环境变量获取）
            model: 要使用的模型（默认为 "gpt-4o"）
        """
        # 获取 API 密钥
        if api_key is None:
            api_key = openai_key
            if not api_key:
                raise ValueError("需要 OpenAI API 密钥。请设置 OPENAI_API_KEY 环境变量或直接传入。")
        
        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(api_key=api_key,
                                    base_url = base_url)
        self.model = model
        
    def chat(self, 
             query: str, 
             system_prompt: str = "你是一个有用的助手，专门解析和理解搜索查询。",
             temperature: float = 0.3) -> str:
        """
        向 GPT-4o 发送查询并获取响应
        
        参数:
            query: 用户查询/搜索项
            system_prompt: 定义助手行为的系统提示
            temperature: 控制随机性（0.0 到 1.0）
            
        返回:
            模型的响应文本
        """
        try:
            # 准备消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # 发送请求到 OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            # 返回响应文本
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"与 GPT-4o 交互时出错: {e}")
            return f"错误: {str(e)}"
    
    def parse_search_query(self, query: str) -> List[str]:
        """
        解析搜索查询并返回关键词列表
        
        参数:
            query: 要解析的搜索查询
            
        返回:
            从查询中提取出的关键词列表
        """
        system_prompt = """
        你是一个专门解析关于女书(nvshu)搜索查询的助手。分析给定的搜索查询，提取关于女书查询重要的关键词。
        
        你的回答应该只包含这些关键词的列表，每个词用逗号分隔。
        不要包含任何解释、标点符号或其他额外文字。
        例如，对于"how to write 女 in nvshu"，你应该返回：
        女
        对于“女书中娘和水怎么写”，你应该返回：
        娘,水
        """
        
        try:
            # 获取模型响应
            response = self.chat(query, system_prompt=system_prompt, temperature=0.1)
            
            # 清理响应
            # 移除可能的代码块标记和额外的标点符号
            cleaned_response = response.strip()
            cleaned_response = re.sub(r'```.*?```', '', cleaned_response, flags=re.DOTALL)
            cleaned_response = re.sub(r'^[^a-zA-Z0-9\u4e00-\u9fff]*|[^a-zA-Z0-9\u4e00-\u9fff]*$', '', cleaned_response)
            
            # 分割成单词列表
            words = [word.strip() for word in cleaned_response.split(',') if word.strip()]
            
            return words
            
        except Exception as e:
            print(f"解析搜索查询时出错: {e}")
            return []

# 示例用法
if __name__ == "__main__":
    # 创建客户端实例
    client = GPT4oClient()
    
    # 简单对话示例
    query = input("请输入您的搜索查询: ")
    result = client.parse_search_query(query)
    
    import json
    print("\n解析结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
