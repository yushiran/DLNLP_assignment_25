#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate a LORA fine-tuning dataset for Nüshu characters
using OpenAI API and the Knowledge Graph.

This script:
1. Reads Nüshu characters from data/processed/data.csv
2. For each character, retrieves context from the Neo4j knowledge graph
3. Uses OpenAI's API to generate 5 question-answer pairs about each character
4. Saves the resulting dataset in JSON format suitable for LORA fine-tuning
"""

import os
import json
import pandas as pd
import logging
import time
from typing import List, Dict, Any
from tqdm import tqdm
import random
from pathlib import Path
from dotenv import load_dotenv
import concurrent.futures
from threading import Lock

# Import the Neo4j graph and OpenAI client
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings, OpenAI
import openai
from gpt4o_search import GPT4oClient

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("lora_dataset_generation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
neo4j_pwd = os.environ.get('NEO4J_PASSWORD')
openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_base_url = os.environ.get('OPENAI_API_BASE_URL', 'https://api.openai.com/v1')
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Template for the RAG prompt
RAG_TEMPLATE = """You are an expert on Nvshu script (女书), a syllabic script used exclusively by women in Jiangyong County, Hunan, China.

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

# Question templates for the dataset
QUESTION_TEMPLATES = [
    "What is the pronunciation of the Nüshu character {character}?",
    "What Chinese character(s) correspond to the Nüshu character {character}?",
    "Can you describe the meaning and usage of the Nüshu character {character}?",
    "How is the Nüshu character {character} written?",
    "In what contexts would the Nüshu character {character} typically be used?",
]

class LoraDatasetGenerator:
    """
    Class to generate a LORA fine-tuning dataset for Nüshu characters
    """
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pwd, openai_api_key, openai_base_url):
        """Initialize the dataset generator with Neo4j and OpenAI credentials"""
        self.openai_client = GPT4oClient(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        
        try:
            # Connect to Neo4j
            self.neo4j_graph = Neo4jGraph(
                url=neo4j_uri, 
                username=neo4j_user, 
                password=neo4j_pwd
            )
            logger.info("Successfully connected to Neo4j database")
            
            # Initialize embedding model
            self.embedding_model = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                base_url=openai_base_url,
                request_timeout=60,
                max_retries=3
            )
            logger.info("Successfully initialized OpenAI embeddings")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    def get_similar_nodes(self, query: str) -> List[Dict[str, Any]]:
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
            retrieval_query,
            params={"embedding": query_embedding}
        )
        return result
    
    def format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for the context"""
        return "\n\n".join([
            f"Source {i+1} ({doc['type']}, similarity: {doc['score']:.2f}):\n{doc['text']}" 
            for i, doc in enumerate(docs)
        ])
    
    def get_context_for_character(self, character: str, chinese_char: str) -> str:
        """Get context from the knowledge graph for a specific Nüshu character"""
        # Combine Nüshu character and Chinese character into a single search query
        search_query = character
        if chinese_char and chinese_char.strip():
            search_query += " " + chinese_char
            
        logger.debug(f"Searching for: {search_query}")
        
        # Get similar nodes using the combined query
        results = self.get_similar_nodes(search_query)
        
        # Take top results (limit to avoid context length issues)
        top_context = results[:5]  # Limit to top 5 results
        
        # Format context for the prompt
        context_text = self.format_docs(top_context)
        return context_text
    
    def generate_qa_pairs(self, character: str, character_info: Dict[str, str], context: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs for a Nüshu character using OpenAI API with multi-threading"""
        # Select 5 random question templates
        selected_templates = random.sample(QUESTION_TEMPLATES, 5)
        
        qa_pairs = []
        # Use ThreadPoolExecutor to process questions in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Create a list to hold the future results
            future_to_template = {
                executor.submit(
                    self.generate_single_qa_pair, 
                    character, 
                    template, 
                    character_info, 
                    context
                ): template for template in selected_templates
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_template):
                template = future_to_template[future]
                try:
                    qa_pair = future.result()
                    if qa_pair:  # Only add if not None
                        logger.debug(f"Successfully generated QA pair for template: {template}")
                        qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.error(f"Error processing template {template} for {character}: {e}")
        
        return qa_pairs
    
    def generate_single_qa_pair(self, character: str, question_template: str, character_info: Dict[str, str], context: str) -> Dict[str, str]:
        """Generate a single question-answer pair for a Nüshu character using OpenAI API"""
        try:
            # Format the question with the character
            question = question_template.format(character=character)
            
            # Create the system prompt
            system_prompt = "You are an expert on Nüshu script (女书), answering questions about specific characters."
            
            # Create the user prompt with context and question
            user_prompt = RAG_TEMPLATE.format(
                context=context,
                question=question
            )
            
            # Call the OpenAI API
            response = self.openai_client.chat(
                query=user_prompt,
                system_prompt=system_prompt
            )
            
            # Create a QA pair in the format needed for LORA fine-tuning
            qa_pair = {
                "Question": question,
                "Context": context,  # Include the context used
                "Response": response
            }
            
            return qa_pair
            
        except Exception as e:
            logger.error(f"Error generating QA pair for question {question_template} about {character}: {e}")
            return None
    
    def process_character(self, character_row) -> List[Dict[str, Any]]:
        """Process a single character and generate QA pairs"""
        try:
            character = character_row['FL Character']
            chinese_char = character_row['Corresponding Chinese Character']
            pronunciation = character_row['Jiangyong Dialect Pronunciation']
            sequence = character_row['《Zitie》Sequence']
            
            character_info = {
                'character': character,
                'chinese_char': chinese_char,
                'pronunciation': pronunciation,
                'sequence': sequence
            }
            
            # Get context for this character
            context = self.get_context_for_character(character, chinese_char)
            logger.debug(f"Retrieved context for {character}: {len(context)} chars")
            
            # Generate QA pairs
            qa_pairs = self.generate_qa_pairs(character, character_info, context)
            logger.debug(f"Generated {len(qa_pairs)} QA pairs for {character}")
            
            # Add character info to each QA pair
            for qa_pair in qa_pairs:
                qa_pair["Character_Info"] = character_info
            
            return qa_pairs
        except Exception as e:
            logger.error(f"Error processing character {character_row.get('FL Character', 'unknown')}: {e}")
            return []
    

    def generate_dataset(self, csv_path: str, output_path: str, limit: int = None, max_workers: int = 3):
        """Generate the LORA fine-tuning dataset from the Nüshu characters CSV with parallel processing"""
        try:
            # Read the Nüshu characters dataset
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} Nüshu characters from {csv_path}")
            
            # Limit the number of characters if specified
            if limit:
                df = df.head(limit)
                logger.info(f"Limited dataset to {limit} characters")
            
            dataset = []
            # Thread-safe result list
            result_lock = Lock()
            
            # Create a shared progress bar
            progress_bar = tqdm(total=len(df), desc="Processing characters")
            
            # Function to update progress bar and add results to dataset
            def process_complete_callback(future):
                # Update progress bar
                progress_bar.update(1)
                
                try:
                    # Get results from the future
                    character_qa_pairs = future.result()
                    
                    # Add to dataset in a thread-safe way
                    with result_lock:
                        dataset.extend(character_qa_pairs)
                except Exception as e:
                    logger.error(f"Error in character processing callback: {e}")
            
            # Process characters in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit jobs for each character
                futures = []
                for _, row in df.iterrows():
                    future = executor.submit(self.process_character, row)
                    future.add_done_callback(process_complete_callback)
                    futures.append(future)
                
                # Wait for all futures to complete (progress updates via callback)
                concurrent.futures.wait(futures)
            
            # Close progress bar
            progress_bar.close()
            
            # Save the dataset to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({'data': dataset}, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully generated dataset with {len(dataset)} QA pairs")
            logger.info(f"Dataset saved to {output_path}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            raise

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = os.path.join(base_dir,"data/processed/lora_dataset") 
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Set paths
    csv_path = os.path.join(base_dir, 'data/processed/data.csv')
    output_path = os.path.join(output_dir,'nushu_lora_dataset.json')
    
    # Initialize the dataset generator
    try:
        generator = LoraDatasetGenerator(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_pwd=neo4j_pwd,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url
        )
        
        # Set parameters for dataset generation
        limit = None  # Limit for testing, set to None for full dataset
        max_workers = 3# Number of parallel workers for character processing
        
        print(f"Starting dataset generation with {max_workers} parallel workers...")
        if limit:
            print(f"Using limit of {limit} characters for testing")
        else:
            print("Generating dataset for all characters")
            
        generator.generate_dataset(csv_path, output_path, limit=limit, max_workers=max_workers)
        
        print(f"✅ Dataset generation complete! Generated LORA fine-tuning data saved to: {output_path}")
        print("To generate the full dataset, set limit=None and adjust max_workers as needed")
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
