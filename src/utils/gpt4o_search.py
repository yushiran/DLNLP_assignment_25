"""
简单的 GPT-4o 客户端

这个脚本提供了一个简单的类，用于与 OpenAI 的 GPT-4o 模型通信，
主要用于解析搜索项和进行简单对话。特别支持女书(Nüshu)字符的处理。
"""

import os
import re
import unicodedata
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
    
    这个类提供方法来发送查询并解析响应，特别适用于女书字符的搜索项解析。
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = openai_base_url, model: str = "gpt-4o-mini"):
        """
        初始化 GPT-4o 客户端
        
        参数:
            api_key: OpenAI API 密钥（如果为 None，将从环境变量获取）
            base_url: OpenAI API 基础URL
            model: 要使用的模型（默认为 "gpt-4o-mini"）
        """
        # 获取 API 密钥
        if api_key is None:
            api_key = openai_key
            if not api_key:
                raise ValueError("需要 OpenAI API 密钥。请设置 OPENAI_API_KEY 环境变量或直接传入。")
        
        # 初始化 OpenAI 客户端
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
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
        解析搜索查询并返回关键词列表，特别支持女书Unicode字符(U+1B170 to U+1B2FF)
        
        参数:
            query: 要解析的搜索查询
            
        返回:
            从查询中提取出的关键词列表
        """
        # 确保查询是UTF-8编码，以正确处理女书字符
        if isinstance(query, str):
            query = unicodedata.normalize('NFC', query)
        
        # 首先检查查询中是否直接包含女书Unicode字符
        nushu_chars = re.findall(r'[\U0001B170-\U0001B2FF]', query)
        if nushu_chars:
            # 如果查询中已有女书字符，直接返回这些字符加上可能的相关关键词
            words = nushu_chars.copy()
            # 添加可能的附加关键词
            if re.search(r'pronunc|pronunciation|发音|读音', query, re.IGNORECASE):
                words.append('pronunciation')
            if re.search(r'mean|meaning|意思|含义', query, re.IGNORECASE):
                words.append('meaning')
            return words
            
        # 增强的系统提示，专门用于处理女书字符相关查询
        system_prompt = """
        你是一个专门解析关于女书(nüshu/nvshu)字符搜索查询的助手。你需要从用户的查询中提取关键词，
        特别注意识别查询中的女书Unicode字符（例如𛅰, 𛅱等）或相关的问题。

        请分析以下几种可能的情况：
        1. 如果查询中出现女书字符（Unicode范围U+1B170至U+1B2FF，如𛅰），直接提取该字符
        2. 如果查询是关于特定女书字符的ID，提取该ID号
        3. 如果查询涉及到某个汉字在女书中的写法，提取该汉字
        4. 如果是关于女书字符发音的问题，提取"pronunciation"和相关字符或ID

        你的回答应该只包含这些关键词的列表，每个词用逗号分隔。
        不要包含任何解释、额外标点符号或其他多余文字。

        示例：
        - 对于"What is the pronunciation of the Nüshu character 𛅰?"，你应该返回：
          𛅰,pronunciation
        - 对于"how to write 女 in nushu"，你应该返回：
          女
        - 对于"女书中娘和水怎么写"，你应该返回：
          娘,水
        - 对于"What does Nüshu character with ID 27 mean?"，你应该返回：
          27,meaning
        """
        
        # 检查是否包含ID数字引用
        id_match = re.search(r'ID\s*[:=]?\s*(\d+)|character\s+(\d+)|字符\s*(\d+)|女书\s*(\d+)', query, re.IGNORECASE)
        if id_match:
            # 提取匹配到的ID号
            id_num = next(filter(None, id_match.groups()))
            words = [id_num]
            
            # 添加可能的附加关键词
            if re.search(r'pronunc|pronunciation|发音|读音', query, re.IGNORECASE):
                words.append('pronunciation')
            if re.search(r'mean|meaning|意思|含义', query, re.IGNORECASE):
                words.append('meaning')
            return words
        
        try:
            # 获取模型响应
            response = self.chat(query, system_prompt=system_prompt, temperature=0.1)
            
            # 清理响应
            # 移除可能的代码块标记和额外的标点符号
            cleaned_response = response.strip()
            cleaned_response = re.sub(r'```.*?```', '', cleaned_response, flags=re.DOTALL)
            
            # 支持更广泛的字符，包括女书Unicode范围
            cleaned_response = re.sub(r'^[^a-zA-Z0-9\u4e00-\u9fff\U0001B170-\U0001B2FF]*|[^a-zA-Z0-9\u4e00-\u9fff\U0001B170-\U0001B2FF]*$', '', cleaned_response)
            
            # 分割成单词列表并保留女书Unicode字符
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
