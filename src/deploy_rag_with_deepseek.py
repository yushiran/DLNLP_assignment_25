"""
Deploying Deepseek RAG with Neo4j Knowledge Graph
This script sets up a Retrieval Augmented Generation system using the Qwen model
and the Nüshu character knowledge graph in Neo4j.
"""

import os
import logging
import streamlit as st
from pathlib import Path
from langchain.callbacks.base import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings  # 替换为 OpenAI 嵌入
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import gpt4o_search
import torch
import time
import urllib.parse
from dotenv import load_dotenv

load_dotenv()
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_pwd = os.getenv("NEO4J_PASSWORD")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.WARNING)  # Suppress Neo4j INFO messages
logger.info(f"neo4j_uri: {neo4j_uri}, neo4j_user: {neo4j_user}, neo4j_pwd: {neo4j_pwd}")

# 模型设置
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 使用Qwen模型替代DeepSeek
# 设置本地模型路径为当前项目下的model目录
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
local_model_path = os.path.join(current_dir, "model", model_id)
cache_dir = os.path.join(current_dir, "model")  # 设置缓存目录为项目下的model目录

# 确保模型目录存在
os.makedirs(local_model_path, exist_ok=True)


class StreamHandler(BaseCallbackHandler):
    """Handler for streaming LLM responses to a Streamlit container"""

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class NushuDeepseekRAG:
    def __init__(self, uri, username, password):
        """
        Initialize Neo4j connection and Deepseek model
        """
        try:
            # Connect to Neo4j
            self.neo4j_graph = Neo4jGraph(url=uri, username=username, password=password)
            logger.info("Successfully connected to Neo4j database")

            # Initialize embedding model with retry mechanism
            try:
                # Make sure we have the API key

                if not openai_api_key:
                    logger.warning("OPENAI_API_KEY not found in environment variables")

                # Attempt to initialize the embedding model with timeout and retries
                self.embedding_model = OpenAIEmbeddings(
                    openai_api_key=openai_api_key,
                    base_url=openai_base_url,
                    request_timeout=60,  # longer timeout
                    max_retries=3,  # retry up to 3 times
                )
                logger.info("Successfully loaded OpenAI embedding model")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {e}")
                raise ValueError(
                    "Could not initialize OpenAI embeddings. Please check your API key and network connection."
                )

            # Initialize Qwen model from local path or download it to local path
            logger.info(f"Loading {model_id} to local path: {local_model_path}...")

            # 检查本地路径是否已存在模型文件
            model_config_path = os.path.join(local_model_path, "config.json")
            if os.path.exists(model_config_path):
                logger.info(
                    f"Model already exists in {local_model_path}, loading from local path"
                )
                model_path_to_use = local_model_path
            else:
                logger.info(
                    f"Model not found in {local_model_path}, downloading model to this location"
                )
                # 将通过设置cache_dir将模型下载到指定路径
                model_path_to_use = model_id

            # 加载tokenizer和模型，设置缓存目录为本地路径
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_to_use, trust_remote_code=True, cache_dir=cache_dir
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_to_use,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            # 如果是从远程下载的，则将下载的模型保存到本地路径
            if model_path_to_use == model_id and not os.path.exists(model_config_path):
                logger.info(f"Saving model to local path: {local_model_path}")
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
                logger.info(f"Model successfully saved to {local_model_path}")

            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                repetition_penalty=1.2,
            )

            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Successfully loaded {model_id} model")

            # Set up chains
            self.setup_chains()

        except Exception as e:
            logger.error(f"Failed to initialize NushuDeepseekRAG: {e}")
            raise

    def setup_chains(self):
        """Set up the RAG chain and direct LLM chain"""
        self.llm_chain = self.configure_llm_only_chain()
        self.rag_chain = self.configure_rag_chain()

    def configure_llm_only_chain(self):
        """Configure a chain that only uses the LLM without retrieval"""
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant specialized in Nüshu script and Chinese characters.
            Answer the question based on your knowledge.
            Question: {question}
            Answer: """
        )
        return prompt | self.llm | StrOutputParser()

    def configure_rag_chain(self):
        """Configure a RAG chain that uses Neo4j vector retrieval"""
        retrieval_query = """
        CALL db.index.vector.queryNodes('nushu_vector_index', 5, $embedding)
        YIELD node, score
        RETURN node.text_repr AS text, score, 
               CASE 
                 WHEN 'NushuCharacter' IN labels(node) THEN 'Nüshu Character' 
                 WHEN 'ChineseCharacter' IN labels(node) THEN 'Chinese Character'
                 WHEN 'EnglishTranslation' IN labels(node) THEN 'English Translation'
                 WHEN 'Pronunciation' IN labels(node) THEN 'Pronunciation'
                 ELSE 'Unknown'
               END AS type
        ORDER BY score DESC
        """

        # 更新提示模板，添加更详细的指导来解析女书字符信息
        rag_prompt = ChatPromptTemplate.from_template(
            """You are an expert on Nvshu script (女书), a syllabic script used exclusively by women in Jiangyong County, Hunan, China.

            Below is retrieved information about Nüshu and Chinese characters. Pay special attention to the format:
            For Nvshu characters: [Nvshu character] [ID number] [corresponding Chinese character(s)] [meaning] [pronunciation]
            
            Retrieved information:
            {context}
            
            Based on the information above, please answer this question: {question}
            
            In your answer:
            1. If searching for a specific character, clearly state the Nüshu character, its ID, and pronunciation
            2. Include the Chinese character correspondence when available
            3. If the exact information isn't found, say so clearly
            
            Be concise and focus only on the information from the retrieved sources."""
        )

        def format_docs(docs):
            return "\n\n".join(
                [
                    f"Source {i+1} ({doc['type']}, similarity: {doc['score']:.2f}):\n{doc['text']}"
                    for i, doc in enumerate(docs)
                ]
            )

        def get_similar_nodes(query):
            # 使用 OpenAI 的嵌入模型生成查询嵌入
            # OpenAI 嵌入模型输出 1536 维度的向量
            query_embedding = self.embedding_model.embed_query(query)
            logger.info(
                f"Generated query embedding with dimension: {len(query_embedding)}"
            )

            # 查询 Neo4j 向量索引
            result = self.neo4j_graph.query(
                retrieval_query, params={"embedding": query_embedding}
            )
            return result

        rag_chain = (
            RunnableParallel(
                {
                    "context": lambda x: format_docs(get_similar_nodes(x["question"])),
                    "question": lambda x: x["question"],
                }
            )
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def get_most_similar_nodes(self, query):
        client = gpt4o_search.GPT4oClient()
        search_terms = client.parse_search_query(query)
        # Step 2: Search for each term in the knowledge graph and collect results
        all_context = []
        for term in search_terms:
            if term:  # Skip empty terms
                logger.info(f"Searching for term: {term}")
                term_results = self._get_similar_nodes(term)
                if term_results:
                    all_context.extend(term_results)
                    logger.info(f"Found {len(term_results)} results for term: {term}")
                else:
                    logger.info(f"No results found for term: {term}")

        # Deduplicate results based on text content
        seen_texts = set()
        unique_context = []
        for result in all_context:
            if result["text"] not in seen_texts:
                seen_texts.add(result["text"])
                unique_context.append(result)

        # Sort by relevance score
        unique_context.sort(key=lambda x: x["score"], reverse=True)

        # Take top results (limit to avoid context length issues)
        top_context = unique_context[:10]
        # Format context for the final prompt
        context_text = self._format_docs(top_context)

        return context_text

    def answer_question(self, question, use_rag=False, stream_handler=None):
        try:
            if use_rag:
                context_text = self.get_most_similar_nodes(
                    question
                )  # Step 1: Retrieve context from Neo4j
                logger.info(f"Retrieved context: {context_text}")

                # Step 3: Answer the original question with retrieved context
                answer_prompt = ChatPromptTemplate.from_template(
                    """You are an expert on Nvshu script (女书), a syllabic script used exclusively by women in Jiangyong County, Hunan, China.

                        Below is retrieved information about Nüshu and Chinese characters. Pay special attention to the format:
                        For Nvshu characters: [Nvshu character] [ID number] [corresponding Chinese character(s)] [meaning] [pronunciation]

                        Retrieved information:
                        {context}

                        Based on the information above, please answer this question: {question}

                        In your answer:
                        1. If searching for a specific character, clearly state the Nüshu character, its ID, and pronunciation
                        2. Include the Chinese character correspondence when available
                        3. If the exact information isn't found, say so clearly

                        Be concise and focus only on the information from the retrieved sources."""
                )

                # Final answer chain
                answer_chain = answer_prompt | self.llm | StrOutputParser()

                # Get the final answer
                result = answer_chain.invoke(
                    {"context": context_text, "question": question}
                )
            else:
                chain = self.llm_chain
                result = chain.invoke({"question": question})

            logger.info(f"question: {question}")
            logger.info(f"result length: {len(result) if result else 0}")

            if not result:
                logger.warning(f"No result returned for the question: {question}")
                return "Sorry, I couldn't find an answer."

            # 根据模型类型处理输出结果
            model_name = model_id.lower()
            answer = self._process_model_response(result, model_name)

            return answer.strip()

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _get_similar_nodes(self, query):
        """Get similar nodes from Neo4j knowledge graph using vector similarity search"""
        retrieval_query = """
        CALL db.index.vector.queryNodes('nushu_vector_index', 5, $embedding)
        YIELD node, score
        RETURN node.text_repr AS text, score, 
               CASE 
                 WHEN 'NushuCharacter' IN labels(node) THEN 'Nüshu Character' 
                 WHEN 'ChineseCharacter' IN labels(node) THEN 'Chinese Character'
                 WHEN 'EnglishTranslation' IN labels(node) THEN 'English Translation'
                 WHEN 'Pronunciation' IN labels(node) THEN 'Pronunciation'
                 ELSE 'Unknown'
               END AS type
        ORDER BY score DESC
        """

        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)

        # Query Neo4j vector index
        result = self.neo4j_graph.query(
            retrieval_query, params={"embedding": query_embedding}
        )
        return result

    def _format_docs(self, docs):
        """Format retrieved documents for display"""
        return "\n\n".join(
            [
                f"Source {i+1} ({doc['type']}, similarity: {doc['score']:.2f}):\n{doc['text']}"
                for i, doc in enumerate(docs)
            ]
        )

    def _process_model_response(self, result, model_name):
        """
        根据不同的模型类型处理返回结果

        Args:
            result: 模型返回的原始结果
            model_name: 模型名称

        Returns:
            str: 处理后的结果
        """
        if not isinstance(result, str):
            return "Sorry, I couldn't parse the response."

        logger.info(f"Processing response from {model_id} model")
        # 处理结果：只保留</think>后的内容
        if "</think>" in result:
            # 只保留</think>后的内容
            answer = result.split("</think>", 1)[-1].strip()
        else:
            answer = result

        return answer


def run_rag_interface():
    """Run the Streamlit interface for the RAG system"""
    st.title(f"Nüshu Knowledge Explorer with {model_id}")
    st.subheader(
        "Ask questions about Nüshu characters, Chinese characters, and their relationships"
    )

    if "rag_system" not in st.session_state:
        with st.spinner(
            f"Initializing RAG system with {model_id} model (this may take a few minutes)..."
        ):
            try:
                st.session_state.rag_system = NushuDeepseekRAG(
                    uri=neo4j_uri, username=neo4j_user, password=neo4j_pwd
                )
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    use_rag = st.sidebar.checkbox("Use RAG (Knowledge Graph Enhanced)", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown(f"**Model:** {model_id}")
    st.sidebar.markdown(f"**RAG Status:** {'Enabled' if use_rag else 'Disabled'}")

    st.sidebar.divider()
    st.sidebar.markdown("### Knowledge Graph")
    st.sidebar.markdown("Nüshu characters: Female-only writing system from China")
    st.sidebar.markdown("Indexed with vector embeddings for semantic search")

    if prompt := st.chat_input("Ask about Nüshu characters or Chinese writing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream_container = st.empty()
            stream_handler = StreamHandler(stream_container)

            try:
                response = st.session_state.rag_system.answer_question(
                    prompt, use_rag=use_rag, stream_handler=stream_handler
                )

                # Add the answer to the chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Display the answer in the stream container if it wasn't already displayed
                if not stream_handler.text:
                    stream_container.markdown(response)

            except Exception as e:
                logger.error(f"Error during assistant response generation: {e}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "An error occurred while generating the response.",
                    }
                )

            st.rerun()


if __name__ == "__main__":

    run_rag_interface()
