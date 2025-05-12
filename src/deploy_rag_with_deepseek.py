"""
Deploying Deepseek RAG with Neo4j Knowledge Graph
This script sets up a Retrieval Augmented Generation system using the Qwen model
and the Nüshu character knowledge graph in Neo4j.
"""

import os
import re
import logging
import streamlit as st
from pathlib import Path
from langchain.callbacks.base import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import gpt4o_search
import torch
import time
import urllib.parse
import locale
import unicodedata
from dotenv import load_dotenv
from utils.unicode_helpers import process_nushu_text, ensure_proper_encoding, remove_repetitive_content

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

# 模型设置
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 使用DeepSeek模型
# 设置本地模型路径为当前项目下的model目录
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_model_path = os.path.join(current_dir, "model", model_id)
finetuned_model_path = os.path.join(current_dir, "model", f"{model_id}-lora-finetuned")
cache_dir = os.path.join(current_dir, "model")  # 设置缓存目录为项目下的model目录

# 确保模型目录存在
os.makedirs(base_model_path, exist_ok=True)
os.makedirs(finetuned_model_path, exist_ok=True)


class StreamHandler(BaseCallbackHandler):
    """Handler for streaming LLM responses to a Streamlit container with enhanced Unicode handling"""

    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.buffer = ""  # Buffer to accumulate tokens for proper Unicode handling
        self.accumulated_text = ""  # Keep track of full accumulated text for processing
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process streaming tokens with proper Unicode handling, specially for Nüshu characters"""
        if not token:
            return
            
        try:
            # Add to buffer for properly handling potentially incomplete Unicode characters
            self.buffer += token
            self.accumulated_text += token
            
            # Check if we need special handling for Nüshu characters
            has_nushu = False
            for codepoint in range(0x1B170, 0x1B300):
                if chr(codepoint) in self.accumulated_text:
                    has_nushu = True
                    break
            
            # Process the accumulated text with our helper functions
            if has_nushu:
                # More careful processing for text with Nüshu characters
                processed_text = process_nushu_text(self.accumulated_text)
            else:
                # Standard encoding handling for other text
                processed_text = ensure_proper_encoding(self.accumulated_text)
            
            # Update the display text - we'll keep the full version for processing later
            self.text = processed_text
            
            # We need to accumulate more characters for proper Unicode handling of multi-byte characters
            # Nüshu characters are particularly complex and can be up to 4 bytes
            if len(self.buffer) >= 8:  # Allow for surrogate pairs and combining characters
                # Clear buffer once we've successfully processed it
                self.buffer = self.buffer[-4:]  # Keep the last 4 characters in case they're incomplete
            
            # Display with proper encoding to ensure character rendering
            # For Nüshu characters, wrap them in a special span for better styling
            display_text = self.text
            if has_nushu:
                # Find and wrap Nüshu characters with special styling
                for match in re.finditer(r'[\U0001B170-\U0001B2FF]', display_text):
                    char = match.group(0)
                    display_text = display_text.replace(char, f'<span class="unicode-fix nushu-character">{char}</span>')
                
                # Use HTML for displaying with custom styles
                self.container.markdown(display_text, unsafe_allow_html=True)
            else:
                # Standard display for regular text
                self.container.markdown(display_text)
                
        except Exception as e:
            # If we encounter an error, it might be due to incomplete Unicode
            # Just accumulate more tokens before displaying
            logger.debug(f"Unicode processing issue: {e}")
            # Continue accumulating without clearing buffer
            
    def on_llm_end(self, *args, **kwargs) -> None:
        """Final processing when LLM generation completes"""
        if self.accumulated_text:
            # Process the complete text to remove any repetitions
            final_text = process_nushu_text(self.accumulated_text)
            self.text = final_text
            # Update the display
            self.container.markdown(self.text)


class NushuDeepseekRAG:
    def __init__(self, uri, username, password, use_finetuned=True):
        """
        Initialize Neo4j connection and Deepseek model
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
            use_finetuned: Whether to use the fine-tuned model
        """
        try:
            # Set the model path based on fine-tuning option
            if use_finetuned:
                self.local_model_path = finetuned_model_path
                logger.info(f"Using fine-tuned model from: {finetuned_model_path}")
            else:
                self.local_model_path = base_model_path
                logger.info(f"Using base model from: {base_model_path}")
                
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
            logger.info(f"Loading model to path: {self.local_model_path}...")

            # 检查本地路径是否已存在模型文件
            model_config_path = os.path.join(self.local_model_path, "config.json")
            if os.path.exists(model_config_path):
                logger.info(
                    f"Model already exists in {self.local_model_path}, loading from local path"
                )
                model_path_to_use = self.local_model_path
            else:
                logger.info(
                    f"Model not found in {self.local_model_path}, downloading model to this location"
                )
                # 将通过设置cache_dir将模型下载到指定路径
                model_path_to_use = model_id

            # 加载tokenizer和模型，设置缓存目录为本地路径
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path_to_use, trust_remote_code=True, cache_dir=cache_dir
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path_to_use,
                torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for better numerical stability
                device_map="auto",
                trust_remote_code=True,
                cache_dir=cache_dir,
                attn_implementation="eager",  # Use eager implementation instead of sdpa to fix the warning
                use_flash_attention_2=False,  # Disable flash attention which can cause issues
                use_cache=True,  # Enable KV cache for better performance
                low_cpu_mem_usage=True  # More memory-efficient loading
            )

            # 如果是从远程下载的，则将下载的模型保存到本地路径
            if model_path_to_use == model_id and not os.path.exists(model_config_path):
                logger.info(f"Saving model to local path: {self.local_model_path}")
                self.tokenizer.save_pretrained(self.local_model_path)
                self.model.save_pretrained(self.local_model_path)
                logger.info(f"Model successfully saved to {self.local_model_path}")

            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=1024,
                # repetition_penalty=1.05,  # Reduced from 1.2 to avoid numerical instability
                # temperature=0.3,  # Add temperature for more stable sampling
                # do_sample=True,   # Use sampling instead of greedy decoding
                # top_k=50,         # Limit to top 50 tokens
                # top_p=0.7        # Use nucleus sampling
            )

            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            logger.info(f"Successfully loaded {self.local_model_path} model")

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
            """<system>
            You are a knowledgeable assistant specializing in the Nüshu writing system from China.
            When responding about Nüshu characters:
            1. Provide CONCISE, NON-REPETITIVE responses
            2. Format information clearly and consistently
            3. If you don't know details about a specific Nüshu character, admit it
            4. Use proper Unicode for any characters you display
            </system>
            
            <question>{question}</question>
            
            Provide a focused answer about Nüshu or Chinese characters based on your knowledge:
            """
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
            
            Be concise and focus only on the information from the retrieved sources.
            """
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
        """
        Search for relevant information in the knowledge graph
        First uses GPT-4o to extract search terms, then searches for each term
        Prioritizes direct matches for Nüshu characters and IDs
        """
        # Step 1: Parse the query to extract key search terms
        client = gpt4o_search.GPT4oClient()
        search_terms = client.parse_search_query(query)
        logger.info(f"Extracted search terms: {search_terms}")
        
        # Step 2: Search for each term in the knowledge graph and collect results
        all_context = []
        direct_match_results = []  # To track results from direct matches 
        
        # First check if query itself contains Nüshu characters or IDs (direct match)
        nushu_chars = re.findall(r'[\U0001B170-\U0001B2FF]', query)
        id_match = re.search(r'ID\s*[:=]?\s*(\d+)|character\s+(\d+)|字符\s*(\d+)|女书\s*(\d+)', query, re.IGNORECASE)
        
        if nushu_chars or (id_match and next(filter(None, id_match.groups()), None)):
            # If the query itself has direct matches, prioritize these results
            logger.info("Query contains direct Nüshu character or ID references")
            direct_results = self._get_similar_nodes(query)
            if direct_results:
                direct_match_results.extend(direct_results)
                direct_match_results = self._format_docs(direct_match_results)
                return direct_match_results
        
        # Then search for each extracted term
        for term in search_terms:
            if term:  # Skip empty terms
                logger.info(f"Searching for term: {term}")
                term_results = self._get_similar_nodes(term)
                if term_results:
                    all_context.extend(term_results)
                    logger.info(f"Found {len(term_results)} results for term: {term}")
                else:
                    logger.info(f"No results found for term: {term}")

        # Combine direct matches (higher priority) with other results
        combined_results = direct_match_results + all_context
        
        # Deduplicate results based on text content
        seen_texts = set()
        unique_context = []
        for result in combined_results:
            if result["text"] not in seen_texts:
                seen_texts.add(result["text"])
                unique_context.append(result)

        # Sort by relevance score and type (Nüshu Characters first)
        def sort_key(result):
            # Type priority: Nüshu Character > Chinese Character > Others
            type_priority = 0
            if result["type"] == "Nüshu Character":
                type_priority = 3
            elif result["type"] == "Chinese Character":
                type_priority = 2
            elif result["type"] == "Pronunciation":
                type_priority = 1
                
            # Combine type priority with score for final sorting
            return (type_priority, result["score"])
            
        unique_context.sort(key=sort_key, reverse=True)

        # Take top results (limit to avoid context length issues)
        top_context = unique_context[:10]
        
        # Format context for the final prompt
        context_text = self._format_docs(top_context)
        
        logger.info(f"Retrieved {len(top_context)} relevant context items")
        return context_text

    def answer_question(self, question, use_rag=False, stream_handler=None):
        try:
            if use_rag:
                context_text = self.get_most_similar_nodes(
                    question
                )  # Step 1: Retrieve context from Neo4j
                logger.info(f"Retrieved context length: {len(context_text)}, preview: {context_text}...")

                # Step 3: Answer the original question with retrieved context
                answer_prompt = ChatPromptTemplate.from_template(
                    """
                    <system>
                    You are a knowledgeable assistant specializing in Nüshu, the women-only writing system from China.
                    When responding about Nüshu characters:
                    1. Provide a SINGLE, CONCISE, and NON-REPETITIVE response
                    2. Format each Nüshu character entry EXACTLY as follows:
                       - Nüshu character: [actual character]
                       - Chinese: [corresponding Chinese character(s)]
                       - Meaning: [meaning]
                       - Pronunciation: [pronunciation]
                    </system>
                    
                    <context>
                    Retrieved information about Nüshu characters and Chinese characters:
                    {context}
                    </context>
                    
                    <question>{question}</question>
                    
                    Provide a single, clear, well-formatted answer based on the retrieved information. Include relevant Nüshu characters with their complete details.
                    """
                )

                answer_chain = answer_prompt | self.llm | StrOutputParser()

                result = answer_chain.invoke(
                    {"context": context_text, "question": question}
                )
                
            else:
                chain = self.llm_chain
                result = chain.invoke({"question": question})

            if not result:
                logger.warning(f"No result returned for the question: {question}")
                return "Sorry, I couldn't find an answer."

            # Process the model output with special attention to Unicode handling
            model_name = model_id.lower()
            answer = self._process_model_response(result, model_name)

            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _get_similar_nodes(self, query):
        """Get similar nodes from Neo4j knowledge graph using exact match first, then vector similarity search"""
        # Check if query is a Nüshu character or contains one (Unicode range U+1B170 to U+1B2FF)
        nushu_chars = re.findall(r'[\U0001B170-\U0001B2FF]', query)
        
        # Check if query contains ID references
        id_match = re.search(r'ID\s*[:=]?\s*(\d+)|character\s+(\d+)|字符\s*(\d+)|女书\s*(\d+)', query, re.IGNORECASE)
        char_id = None
        if id_match:
            # Get the first non-None group as the character ID
            char_id = next(filter(None, id_match.groups()))
        
        # If we found exact Nüshu characters or IDs, use direct matching first
        if nushu_chars or char_id:
            logger.info(f"Using direct match for: {nushu_chars if nushu_chars else 'ID: ' + char_id}")
            
            # Prepare direct match query with more comprehensive matching
            direct_match_query = """
            MATCH (n)
            WHERE
            """
            
            # Add conditions based on what we found
            conditions = []
            params = {}
            
            if nushu_chars:
                # Add condition for each Nüshu character with multiple match options
                for i, char in enumerate(nushu_chars):
                    param_name = f"char_{i}"
                    # Match the character in various fields where it might appear
                    # Use 'IS NOT NULL' syntax instead of EXISTS() as per Neo4j requirement
                    char_conditions = [
                        f"n.character = ${param_name}",
                        f"n.text_repr CONTAINS ${param_name}",
                        f"n.text IS NOT NULL AND n.text CONTAINS ${param_name}"
                    ]
                    conditions.append("(" + " OR ".join(char_conditions) + ")")
                    params[param_name] = char
            
            if char_id:
                # Add condition for character ID with various formats
                id_conditions = [
                    "n.id = $char_id",
                    "toString(n.id) = $char_id_str",
                    "n.text_repr CONTAINS $id_text"
                ]
                conditions.append("(" + " OR ".join(id_conditions) + ")")
                params["char_id"] = char_id
                params["char_id_str"] = str(char_id)
                params["id_text"] = f"ID {char_id}"
            
            # Combine conditions with OR
            direct_match_query += " OR ".join(conditions)
            
            direct_match_query += """
            RETURN n.text_repr AS text, 1.0 AS score,
                   CASE 
                     WHEN 'NushuCharacter' IN labels(n) THEN 'Nüshu Character' 
                     WHEN 'ChineseCharacter' IN labels(n) THEN 'Chinese Character'
                     WHEN 'EnglishTranslation' IN labels(n) THEN 'English Translation'
                     WHEN 'Pronunciation' IN labels(n) THEN 'Pronunciation'
                     ELSE 'Unknown'
                   END AS type
            ORDER BY 
                CASE 
                    WHEN 'NushuCharacter' IN labels(n) THEN 0 
                    WHEN 'ChineseCharacter' IN labels(n) THEN 1
                    ELSE 2
                END
            LIMIT 10
            """
            
            # Execute direct match query
            try:
                direct_results = self.neo4j_graph.query(direct_match_query, params=params)
                
                # If we found exact matches, return them
                if direct_results:
                    logger.info(f"Found {len(direct_results)} direct matches")
                    
                    return direct_results
                    
                # Otherwise continue to embedding search
                logger.info("No direct matches found, falling back to embedding search")
            except Exception as e:
                logger.warning(f"Direct match query failed: {e}, falling back to embedding search")
        
        # Standard vector similarity search as fallback
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

        # Generate query embedding for semantic search
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
        
    def reload_model_with_variant(self, use_finetuned=False):
        """
        Reload the model with either base or fine-tuned variant
        
        Args:
            use_finetuned: Whether to use the fine-tuned model
        """
        # Clear CUDA cache to free memory
        torch.cuda.empty_cache()
        
        # Set the model path based on fine-tuning option
        if use_finetuned:
            self.local_model_path = finetuned_model_path
            logger.info(f"Reloading with fine-tuned model from: {finetuned_model_path}")
        else:
            self.local_model_path = base_model_path
            logger.info(f"Reloading with base model from: {base_model_path}")
        
        # Load the tokenizer and model
        model_config_path = os.path.join(self.local_model_path, "config.json")
        if os.path.exists(model_config_path):
            model_path_to_use = self.local_model_path
        else:
            model_path_to_use = model_id
            
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path_to_use, trust_remote_code=True, cache_dir=cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_to_use,
            torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for better numerical stability
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="eager",  # Use eager implementation instead of sdpa to fix the warning
            use_flash_attention_2=False,  # Disable flash attention which can cause issues
            use_cache=True,  # Enable KV cache for better performance
            low_cpu_mem_usage=True  # More memory-efficient loading
        )
        
        # Create a new pipeline with carefully optimized parameters to prevent repetition and handle Unicode
        text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=800,         # Reduced for more concise outputs
            repetition_penalty=1.2,     # Increased to more aggressively prevent repetition
            temperature=0.65,           # Moderate temperature for balanced output
            do_sample=True,             # Enable sampling for more varied responses
            top_k=40,                   # Limit to top 40 tokens for more focused output
            top_p=0.85,                 # Allow for varied responses while maintaining coherence
            no_repeat_ngram_size=4,     # Increased to prevent longer repetitive patterns
            bad_words_ids=None,         # Don't explicitly block any tokens
            force_words_ids=None,       # Don't force any specific tokens
            renormalize_logits=True,    # Normalize logits after modifications like repetition penalty
            use_cache=True              # Use KV cache for better performance
        )
        
        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        logger.info(f"Successfully reloaded {'fine-tuned' if use_finetuned else 'base'} model")
        
        # Set up chains
        self.setup_chains()
        
        return True

    def _process_model_response(self, result, model_name):
        """
        Process the model's response based on model type and ensure proper character rendering
        
        Args:
            result: Raw model response
            model_name: Model name/identifier
            
        Returns:
            str: Processed response with correctly rendered characters
        """
        if not isinstance(result, str):
            return "Sorry, I couldn't parse the response."
            
        logger.info(f"Model response length: {len(result)}， preview: {result}...")
        
        # Extract main content (remove any system thinking)
        if "</think>" in result:
            answer = result.split("</think>", 1)[-1].strip()
        elif "<question>" in result and "</question>" in result:
            # Extract only the answer part after the question
            answer = result.split("</question>", 1)[-1].strip()
        elif "<system>" in result and "</system>" in result:
            # Extract only the content after system instructions
            answer = result.split("</system>", 1)[-1].strip()
        else:
            answer = result
        
        logger.info(f"Processed answer length: {len(answer)}， preview: {answer}")
        return answer

def run_rag_interface():
    """Run the Streamlit interface for the RAG system"""
    st.title(f"Nüshu Knowledge Explorer with {model_id}")
    st.subheader(
        "Ask questions about Nüshu characters, Chinese characters, and their relationships"
    )

    # Add model selection checkbox in the sidebar
    use_finetuned = st.sidebar.checkbox("Use Fine-tuned Model", value=False)
    
    # Initialize the RAG system if it doesn't exist
    if "rag_system" not in st.session_state:
        with st.spinner(
            f"Initializing RAG system with {'fine-tuned' if use_finetuned else 'base'} {model_id} model (this may take a few minutes)..."
        ):
            try:
                st.session_state.rag_system = NushuDeepseekRAG(
                    uri=neo4j_uri, username=neo4j_user, password=neo4j_pwd,
                    use_finetuned=use_finetuned
                )
                st.session_state.use_finetuned = use_finetuned
                st.success(f"RAG system initialized successfully with {'fine-tuned' if use_finetuned else 'base'} model!")
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
                return
    # Check if we need to switch models
    elif "use_finetuned" in st.session_state and st.session_state.use_finetuned != use_finetuned:
        with st.spinner(f"Switching to {'fine-tuned' if use_finetuned else 'base'} model..."):
            try:
                st.session_state.rag_system.reload_model_with_variant(use_finetuned=use_finetuned)
                st.session_state.use_finetuned = use_finetuned
                st.success(f"Successfully switched to {'fine-tuned' if use_finetuned else 'base'} model!")
            except Exception as e:
                st.error(f"Failed to switch models: {e}")
                return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    use_rag = st.sidebar.checkbox("Use RAG (Knowledge Graph Enhanced)", value=True)

    st.sidebar.divider()
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown(f"**Base Model:** {model_id}")
    st.sidebar.markdown(f"**Using Fine-tuned Model:** {'Yes' if use_finetuned else 'No'}")
    st.sidebar.markdown(f"**RAG Status:** {'Enabled' if use_rag else 'Disabled'}")

    st.sidebar.divider()
    st.sidebar.markdown("### Knowledge Graph")
    st.sidebar.markdown("Nüshu characters: Female-only writing system from China")
    st.sidebar.markdown("Indexed with vector embeddings for semantic search")
    
    st.sidebar.divider()
    st.sidebar.markdown("### Nüshu Character Display")
    st.sidebar.markdown("If you see squares or � symbols instead of characters, your browser may not support Nüshu Unicode rendering.")
    st.sidebar.info("Try asking about specific Nüshu characters by their ID numbers if the characters don't display correctly.")
    
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
