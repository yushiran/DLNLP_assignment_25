"""
Adding Vector Index to the Knowledge Graph
This script enhances the Nüshu knowledge graph with vector embeddings
to enable semantic search capabilities for all node types.
"""
import os
import logging
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings import
import openai  # openai import
load_dotenv()

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.WARNING)  # Suppress Neo4j INFO messages

class NushuVectorIndex:
    def __init__(self, uri, username, password, batch_size=32):
        """
        Initialize Neo4j connection
        
        Parameters:
        - uri: Neo4j database URI
        - username: Neo4j username
        - password: Neo4j password
        - batch_size: Number of embeddings to generate and store at once
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("Successfully connected to Neo4j database")
            
            # Check if OPENAI_API_KEY is set
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            if not openai_api_key:
                logger.error("OPENAI_API_KEY environment variable is not set.")
                logger.error("Please set your OpenAI API key using: export OPENAI_API_KEY=your_api_key")
                raise ValueError("OpenAI API key is required")
                
            # Using OpenAI embeddings with 1536 dimensions
            logger.info("Using OpenAI embeddings (1536 dimensions)")
            self.embeddings = OpenAIEmbeddings()
            self.dimension = 1536
            self.batch_size = batch_size
            logger.info(f"OpenAI embedding model loaded successfully (batch size: {batch_size})")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def close(self):
        """
        Close Neo4j connection
        """
        self.driver.close()
        logger.info("Neo4j connection closed")

    def create_vector_indices(self, drop_existing=True):
        """
        Create vector indices for all node types in Neo4j
        
        Parameters:
        - drop_existing: If True, drop existing vector indices before creating new ones
        """
        with self.driver.session() as session:
            # Define node types and their corresponding index names
            node_types = [
                ("NushuCharacter", "nushu_vector_index"),
                ("ChineseCharacter", "chinese_vector_index"),
                ("EnglishTranslation", "english_vector_index"),
                ("Pronunciation", "pronunciation_vector_index")
            ]
            
            for node_type, index_name in node_types:
                # Check if vector index exists
                result = session.run("""
                    SHOW INDEXES YIELD name, type
                    WHERE name = $index_name AND type = 'VECTOR'
                    RETURN count(*) > 0 AS exists
                """, index_name=index_name)
                index_exists = result.single()["exists"]
                
                if index_exists and drop_existing:
                    # Drop existing vector index
                    logger.info(f"Dropping existing vector index for {node_type}...")
                    session.run(f"""
                        DROP INDEX {index_name} IF EXISTS
                    """)
                    logger.info(f"Existing vector index for {node_type} dropped")
                
                if not index_exists or drop_existing:
                    # Create vector index for the node type with 1536 dimensions
                    logger.info(f"Creating vector index for {node_type} nodes with {self.dimension} dimensions...")
                    session.run(f"""
                        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                        FOR (n:{node_type}) 
                        ON (n.embedding)
                        OPTIONS {{indexConfig: {{
                          `vector.dimensions`: {self.dimension},
                          `vector.similarity_function`: 'cosine'
                        }}}}
                    """)
                    logger.info(f"Vector index for {node_type} created successfully with {self.dimension} dimensions")
                else:
                    logger.info(f"Vector index for {node_type} already exists and was kept as is")

    def generate_embeddings(self, texts):
        """
        Generate embeddings using OpenAI API
        
        Parameters:
        - texts: List of strings to generate embeddings for
        
        Returns:
        - List of embeddings as Python lists
        """
        results = self.embeddings.embed_documents(texts)
        return results

    def generate_embeddings_for_nushu_characters(self):
        """
        Generate embeddings for all Nüshu characters with relationship context
        using batch processing
        """
        with self.driver.session() as session:
            # Get all Nüshu characters with their properties and relationships
            result = session.run("""
                MATCH (n:NushuCharacter)-[r1:CORRESPONDS_TO]->(c:ChineseCharacter)-[r2:TRANSLATES_TO]->(e:EnglishTranslation)
                MATCH (n)-[r3:PRONOUNCED_AS]->(p:Pronunciation)
                RETURN n.character as nushu, n.sequence as sequence, 
                       collect(distinct c.character) as chinese, 
                       collect(distinct e.text) as english, 
                       p.value as pronunciation
            """)
            
            records = list(result)
            total_records = len(records)
            logger.info(f"Found {total_records} Nüshu characters to process")
            
            batch = []
            batch_texts = []
            batch_ids = []
            
            for i, record in enumerate(records):
                nushu_char = record["nushu"]
                sequence = record["sequence"]
                chinese_chars = " ".join(record["chinese"])
                english_texts = " ".join(record["english"])
                pronunciation = record["pronunciation"]
                
                # Create a text representation without descriptive labels
                text_repr = f"{nushu_char} {sequence} {chinese_chars} {english_texts} {pronunciation}"
                
                batch.append((nushu_char, text_repr))
                batch_texts.append(text_repr)
                batch_ids.append(nushu_char)
                
                # Process batch when it reaches batch_size or at the end
                if len(batch) >= self.batch_size or i == total_records - 1:
                    # Generate embeddings for the batch
                    embeddings = self.generate_embeddings(batch_texts)
                    
                    # Create parameter list for batch update
                    params = []
                    for j in range(len(batch)):
                        params.append({
                            "nushu": batch_ids[j],
                            "embedding": embeddings[j],
                            "text_repr": batch_texts[j]
                        })
                    
                    # Store embeddings in Neo4j
                    session.run("""
                        UNWIND $params AS param
                        MATCH (n:NushuCharacter {character: param.nushu})
                        SET n.embedding = param.embedding
                        SET n.text_repr = param.text_repr
                    """, params=params)
                    
                    logger.info(f"Generated embeddings for {min(i+1, total_records)}/{total_records} Nüshu characters")
                    
                    # Clear batch
                    batch = []
                    batch_texts = []
                    batch_ids = []
            
            logger.info(f"Completed generating embeddings for {total_records} Nüshu characters")

    def generate_embeddings_for_chinese_characters(self):
        """
        Generate embeddings for all Chinese characters with relationship context
        using batch processing
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:ChineseCharacter)<-[:CORRESPONDS_TO]-(n:NushuCharacter)
                MATCH (c)-[:TRANSLATES_TO]->(e:EnglishTranslation)
                MATCH (n)-[:PRONOUNCED_AS]->(p:Pronunciation)
                RETURN c.character as chinese, 
                       collect(distinct n.character) as nushu, 
                       collect(distinct e.text) as english,
                       collect(distinct p.value) as pronunciations
            """)
            
            records = list(result)
            total_records = len(records)
            logger.info(f"Found {total_records} Chinese characters to process")
            
            batch = []
            batch_texts = []
            batch_ids = []
            
            for i, record in enumerate(records):
                chinese_char = record["chinese"]
                nushu_chars = " ".join(record["nushu"])
                english_texts = " ".join(record["english"])
                pronunciations = " ".join(record["pronunciations"])
                
                # Create a text representation without descriptive labels
                text_repr = f"{chinese_char} {nushu_chars} {english_texts} {pronunciations}"
                
                batch.append((chinese_char, text_repr))
                batch_texts.append(text_repr)
                batch_ids.append(chinese_char)
                
                # Process batch when it reaches batch_size or at the end
                if len(batch) >= self.batch_size or i == total_records - 1:
                    # Generate embeddings for the batch
                    embeddings = self.generate_embeddings(batch_texts)
                    
                    # Create parameter list for batch update
                    params = []
                    for j in range(len(batch)):
                        params.append({
                            "chinese": batch_ids[j],
                            "embedding": embeddings[j],
                            "text_repr": batch_texts[j]
                        })
                    
                    # Store embeddings in Neo4j
                    session.run("""
                        UNWIND $params AS param
                        MATCH (c:ChineseCharacter {character: param.chinese})
                        SET c.embedding = param.embedding
                        SET c.text_repr = param.text_repr
                    """, params=params)
                    
                    logger.info(f"Generated embeddings for {min(i+1, total_records)}/{total_records} Chinese characters")
                    
                    # Clear batch
                    batch = []
                    batch_texts = []
                    batch_ids = []
            
            logger.info(f"Completed generating embeddings for {total_records} Chinese characters")

    def generate_embeddings_for_english_translations(self):
        """
        Generate embeddings for all English translations with relationship context
        using batch processing
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:EnglishTranslation)<-[:TRANSLATES_TO]-(c:ChineseCharacter)<-[:CORRESPONDS_TO]-(n:NushuCharacter)
                RETURN e.text as english, 
                       collect(distinct c.character) as chinese, 
                       collect(distinct n.character) as nushu
            """)
            
            records = list(result)
            total_records = len(records)
            logger.info(f"Found {total_records} English translations to process")
            
            batch = []
            batch_texts = []
            batch_ids = []
            
            for i, record in enumerate(records):
                english_text = record["english"]
                chinese_chars = " ".join(record["chinese"])
                nushu_chars = " ".join(record["nushu"])
                
                # Create a text representation without descriptive labels
                text_repr = f"{english_text} {chinese_chars} {nushu_chars}"
                
                batch.append((english_text, text_repr))
                batch_texts.append(text_repr)
                batch_ids.append(english_text)
                
                # Process batch when it reaches batch_size or at the end
                if len(batch) >= self.batch_size or i == total_records - 1:
                    # Generate embeddings for the batch
                    embeddings = self.generate_embeddings(batch_texts)
                    
                    # Create parameter list for batch update
                    params = []
                    for j in range(len(batch)):
                        params.append({
                            "english": batch_ids[j],
                            "embedding": embeddings[j],
                            "text_repr": batch_texts[j]
                        })
                    
                    # Store embeddings in Neo4j
                    session.run("""
                        UNWIND $params AS param
                        MATCH (e:EnglishTranslation {text: param.english})
                        SET e.embedding = param.embedding
                        SET e.text_repr = param.text_repr
                    """, params=params)
                    
                    logger.info(f"Generated embeddings for {min(i+1, total_records)}/{total_records} English translations")
                    
                    # Clear batch
                    batch = []
                    batch_texts = []
                    batch_ids = []
            
            logger.info(f"Completed generating embeddings for {total_records} English translations")

    def generate_embeddings_for_pronunciations(self):
        """
        Generate embeddings for all pronunciations with relationship context
        using batch processing
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Pronunciation)<-[:PRONOUNCED_AS]-(n:NushuCharacter)-[:CORRESPONDS_TO]->(c:ChineseCharacter)
                RETURN p.value as pronunciation, 
                       collect(distinct n.character) as nushu, 
                       collect(distinct c.character) as chinese
            """)
            
            records = list(result)
            total_records = len(records)
            logger.info(f"Found {total_records} pronunciations to process")
            
            batch = []
            batch_texts = []
            batch_ids = []
            
            for i, record in enumerate(records):
                pronunciation = record["pronunciation"]
                nushu_chars = " ".join(record["nushu"])
                chinese_chars = " ".join(record["chinese"])
                
                # Create a text representation without descriptive labels
                text_repr = f"{pronunciation} {nushu_chars} {chinese_chars}"
                
                batch.append((pronunciation, text_repr))
                batch_texts.append(text_repr)
                batch_ids.append(pronunciation)
                
                # Process batch when it reaches batch_size or at the end
                if len(batch) >= self.batch_size or i == total_records - 1:
                    # Generate embeddings for the batch
                    embeddings = self.generate_embeddings(batch_texts)
                    
                    # Create parameter list for batch update
                    params = []
                    for j in range(len(batch)):
                        params.append({
                            "pronunciation": batch_ids[j],
                            "embedding": embeddings[j],
                            "text_repr": batch_texts[j]
                        })
                    
                    # Store embeddings in Neo4j
                    session.run("""
                        UNWIND $params AS param
                        MATCH (p:Pronunciation {value: param.pronunciation})
                        SET p.embedding = param.embedding
                        SET p.text_repr = param.text_repr
                    """, params=params)
                    
                    logger.info(f"Generated embeddings for {min(i+1, total_records)}/{total_records} pronunciations")
                    
                    # Clear batch
                    batch = []
                    batch_texts = []
                    batch_ids = []
            
            logger.info(f"Completed generating embeddings for {total_records} pronunciations")

    def test_semantic_search(self, query, node_type, top_k=5):
        """
        Test semantic search functionality across different node types
        
        Parameters:
        - query: The search query text
        - node_type: The type of node to search ('NushuCharacter', 'ChineseCharacter', 
                    'EnglishTranslation', or 'Pronunciation')
        - top_k: Number of results to return
        """
        # Map node types to their index names
        index_map = {
            'NushuCharacter': 'nushu_vector_index',
            'ChineseCharacter': 'chinese_vector_index',
            'EnglishTranslation': 'english_vector_index',
            'Pronunciation': 'pronunciation_vector_index'
        }
        
        # Generate query embedding using OpenAI
        query_embedding = self.generate_embeddings([query])[0]
        
        with self.driver.session() as session:
            # Construct the appropriate Cypher query based on node type
            if node_type == 'NushuCharacter':
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    MATCH (node)-[:CORRESPONDS_TO]->(c:ChineseCharacter)
                    MATCH (node)-[:PRONOUNCED_AS]->(p:Pronunciation)
                    RETURN node.character as nushu, node.text_repr as text_repr,
                           collect(distinct c.character) as chinese, p.value as pronunciation, score
                    ORDER BY score DESC
                """, index_name=index_map[node_type], top_k=top_k, embedding=query_embedding)
            elif node_type == 'ChineseCharacter':
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    MATCH (node)<-[:CORRESPONDS_TO]-(n:NushuCharacter)
                    MATCH (node)-[:TRANSLATES_TO]->(e:EnglishTranslation)
                    RETURN node.character as chinese, node.text_repr as text_repr,
                           collect(distinct n.character) as nushu, 
                           collect(distinct e.text) as english, score
                    ORDER BY score DESC
                """, index_name=index_map[node_type], top_k=top_k, embedding=query_embedding)
            elif node_type == 'EnglishTranslation':
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    MATCH (node)<-[:TRANSLATES_TO]-(c:ChineseCharacter)<-[:CORRESPONDS_TO]-(n:NushuCharacter)
                    RETURN node.text as english, node.text_repr as text_repr,
                           collect(distinct c.character) as chinese, 
                           collect(distinct n.character) as nushu, score
                    ORDER BY score DESC
                """, index_name=index_map[node_type], top_k=top_k, embedding=query_embedding)
            elif node_type == 'Pronunciation':
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    MATCH (node)<-[:PRONOUNCED_AS]-(n:NushuCharacter)-[:CORRESPONDS_TO]->(c:ChineseCharacter)
                    RETURN node.value as pronunciation, node.text_repr as text_repr,
                           collect(distinct n.character) as nushu, 
                           collect(distinct c.character) as chinese, score
                    ORDER BY score DESC
                """, index_name=index_map[node_type], top_k=top_k, embedding=query_embedding)
            else:
                logger.error(f"Invalid node type: {node_type}")
                return
            
            logger.info(f"Search results for query: '{query}' in node type: {node_type}")
            logger.info("-" * 50)
            
            records = list(result)
            if not records:
                logger.info("No results found")
                return
                
            for record in records:
                if node_type == 'NushuCharacter':
                    logger.info(f"Nushu character: {record['nushu']}")
                    logger.info(f"Corresponding Chinese: {', '.join(record['chinese'])}")
                    logger.info(f"Pronunciation: {record['pronunciation']}")
                elif node_type == 'ChineseCharacter':
                    logger.info(f"Chinese character: {record['chinese']}")
                    logger.info(f"Corresponding Nushu: {', '.join(record['nushu'])}")
                    logger.info(f"English translation: {', '.join(record['english'])}")
                elif node_type == 'EnglishTranslation':
                    logger.info(f"English: {record['english']}")
                    logger.info(f"Corresponding Chinese: {', '.join(record['chinese'])}")
                    logger.info(f"Corresponding Nushu: {', '.join(record['nushu'])}")
                elif node_type == 'Pronunciation':
                    logger.info(f"Pronunciation: {record['pronunciation']}")
                    logger.info(f"Corresponding Nushu: {', '.join(record['nushu'])}")
                    logger.info(f"Corresponding Chinese: {', '.join(record['chinese'])}")
                
                logger.info(f"Text: {record['text_repr']}")
                logger.info(f"Similarity score: {record['score']}")
                logger.info("-" * 50)

    def cross_node_semantic_search(self, query, top_k=3):
        """
        Perform semantic search across all node types and show combined results
        
        This function searches all node types and returns the top results
        from each type, ordered by similarity score.
        """
        # Generate query embedding using OpenAI
        query_embedding = self.generate_embeddings([query])[0]
        
        combined_results = []
        node_types = ['NushuCharacter', 'ChineseCharacter', 'EnglishTranslation', 'Pronunciation']
        index_map = {
            'NushuCharacter': 'nushu_vector_index',
            'ChineseCharacter': 'chinese_vector_index',
            'EnglishTranslation': 'english_vector_index',
            'Pronunciation': 'pronunciation_vector_index'
        }
        
        with self.driver.session() as session:
            for node_type in node_types:
                # Execute search for each node type
                result = session.run("""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    RETURN node, labels(node)[0] AS type, score
                    ORDER BY score DESC
                """, index_name=index_map[node_type], top_k=top_k, embedding=query_embedding)
                
                for record in result:
                    node = record["node"]
                    node_type = record["type"]
                    score = record["score"]
                    
                    if node_type == 'NushuCharacter':
                        identifier = node["character"]
                    elif node_type == 'ChineseCharacter':
                        identifier = node["character"]
                    elif node_type == 'EnglishTranslation':
                        identifier = node["text"]
                    elif node_type == 'Pronunciation':
                        identifier = node["value"]
                    
                    combined_results.append({
                        "type": node_type,
                        "identifier": identifier,
                        "text_repr": node.get("text_repr", ""),
                        "score": score
                    })
            
            # Sort combined results by score
            combined_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Print results
            logger.info(f"Cross-node search results for query: '{query}'")
            logger.info("-" * 50)
            
            for result in combined_results:
                logger.info(f"Node type: {result['type']}")
                logger.info(f"Identifier: {result['identifier']}")
                logger.info(f"Text representation: {result['text_repr']}")
                logger.info(f"Similarity score: {result['score']}")
                logger.info("-" * 50)

def main():
    try:
        # Get Neo4j connection details
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
        neo4j_pwd = os.environ.get('NEO4J_PASSWORD')
        
        # If password is not in env, try to read from file
        if not neo4j_pwd:
            try:
                with open(os.path.join(base_dir, 'Neo4j-pwd.txt'), 'r') as f:
                    neo4j_pwd = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read Neo4j password from file: {e}")
                raise
        
        # Initialize vector index class with OpenAI embeddings
        vector_index = NushuVectorIndex(neo4j_uri, neo4j_user, neo4j_pwd, batch_size=256)
        
        # Create vector indices with 1536 dimensions, dropping any existing indices
        vector_index.create_vector_indices(drop_existing=True)
        
        # Generate embeddings for all node types using OpenAI
        vector_index.generate_embeddings_for_nushu_characters()
        vector_index.generate_embeddings_for_chinese_characters()
        vector_index.generate_embeddings_for_english_translations()
        vector_index.generate_embeddings_for_pronunciations()
        
        # Test semantic search functionality for different node types
        logger.info("\nNushu character search test:")
        vector_index.test_semantic_search("woman", 'NushuCharacter', 3)
        
        logger.info("\nChinese character search test:")
        vector_index.test_semantic_search("water", 'ChineseCharacter', 3)
        
        logger.info("\nEnglish translation search test:")
        vector_index.test_semantic_search("家", 'EnglishTranslation', 3)
        
        logger.info("\nPronunciation search test:")
        vector_index.test_semantic_search("tian", 'Pronunciation', 3)
        
        logger.info("\nCross-node type search test:")
        vector_index.cross_node_semantic_search("水", 2)
        
        # Close connection
        vector_index.close()
        
        logger.info("Vector indexing and testing complete for all node types")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()