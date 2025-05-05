"""
Adding Vector Index to the Knowledge Graph
This script enhances the Nüshu knowledge graph with vector embeddings
to enable semantic search capabilities.
"""
import os
import logging
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("neo4j").setLevel(logging.WARNING)  # Suppress Neo4j INFO messages

class NushuVectorIndex:
    def __init__(self, uri, username, password):
        """
        Initialize Neo4j connection
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("Successfully connected to Neo4j database")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Successfully loaded sentence transformer model")
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    def close(self):
        """
        Close Neo4j connection
        """
        self.driver.close()
        logger.info("Neo4j connection closed")

    def create_vector_index(self):
        """
        Create vector index in Neo4j
        """
        with self.driver.session() as session:
            # Check if vector index exists
            result = session.run("""
                SHOW INDEXES YIELD name, type
                WHERE name = 'nushu_vector_index' AND type = 'VECTOR'
                RETURN count(*) > 0 AS exists
            """)
            index_exists = result.single()["exists"]
            
            if not index_exists:
                # Create vector index for NushuCharacter nodes
                logger.info("Creating vector index for NushuCharacter nodes...")
                session.run("""
                    CREATE VECTOR INDEX nushu_vector_index IF NOT EXISTS
                    FOR (n:NushuCharacter) 
                    ON (n.embedding)
                    OPTIONS {indexConfig: {
                      `vector.dimensions`: 384,
                      `vector.similarity_function`: 'cosine'
                    }}
                """)
                logger.info("Vector index created successfully")
            else:
                logger.info("Vector index already exists")

    def generate_embeddings_for_nushu_characters(self):
        """
        Generate embeddings for all Nüshu characters and store them in Neo4j
        """
        with self.driver.session() as session:
            # Get all Nüshu characters with their properties
            result = session.run("""
                MATCH (n:NushuCharacter)-[:CORRESPONDS_TO]->(c:ChineseCharacter)-[:TRANSLATES_TO]->(e:EnglishTranslation)
                MATCH (n)-[:PRONOUNCED_AS]->(p:Pronunciation)
                RETURN n.character as nushu, collect(c.character) as chinese, 
                       collect(e.text) as english, p.value as pronunciation
            """)
            
            count = 0
            for record in result:
                nushu_char = record["nushu"]
                chinese_chars = " ".join(record["chinese"])
                english_texts = " ".join(record["english"])
                pronunciation = record["pronunciation"]
                
                # Create a text representation for embedding
                text_repr = f"{chinese_chars} {english_texts} {pronunciation}"
                
                # Generate embedding
                embedding = self.model.encode(text_repr).tolist()
                
                # Store embedding in Neo4j
                session.run("""
                    MATCH (n:NushuCharacter {character: $nushu})
                    SET n.embedding = $embedding
                    SET n.text_repr = $text_repr
                """, nushu=nushu_char, embedding=embedding, text_repr=text_repr)
                
                count += 1
                if count % 20 == 0:
                    logger.info(f"Generated embeddings for {count} Nüshu characters")
            
            logger.info(f"Completed generating embeddings for {count} Nüshu characters")

    def test_vector_search(self, query, top_k=5):
        """
        Test vector search functionality
        """
        # Generate query embedding
        query_embedding = self.model.encode(query).tolist()
        
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('nushu_vector_index', $top_k, $embedding)
                YIELD node, score
                MATCH (node)-[:CORRESPONDS_TO]->(c:ChineseCharacter)
                MATCH (node)-[:PRONOUNCED_AS]->(p:Pronunciation)
                RETURN node.character as nushu, node.text_repr as text_repr,
                       collect(c.character) as chinese, p.value as pronunciation, score
                ORDER BY score DESC
            """, top_k=top_k, embedding=query_embedding)
            
            logger.info(f"Search results for query: '{query}'")
            logger.info("-" * 50)
            
            records = list(result)
            if not records:
                logger.info("No results found")
                return
                
            for record in records:
                logger.info(f"Nüshu: {record['nushu']}")
                logger.info(f"Chinese: {', '.join(record['chinese'])}")
                logger.info(f"Pronunciation: {record['pronunciation']}")
                logger.info(f"Text: {record['text_repr']}")
                logger.info(f"Similarity score: {record['score']}")
                logger.info("-" * 50)

def main():
    try:
        # Get Neo4j connection details
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        neo4j_pwd = os.environ.get('NEO4J_PASSWORD')
        
        # If password is not in env, try to read from file
        if not neo4j_pwd:
            try:
                with open(os.path.join(base_dir, 'Neo4j-pwd.txt'), 'r') as f:
                    neo4j_pwd = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read Neo4j password from file: {e}")
                raise
        
        # Initialize vector index class
        vector_index = NushuVectorIndex(neo4j_uri, neo4j_user, neo4j_pwd)
        
        # Create vector index
        vector_index.create_vector_index()
        
        # Generate embeddings for all Nüshu characters
        vector_index.generate_embeddings_for_nushu_characters()
        
        # Test vector search functionality
        vector_index.test_vector_search("woman", 5)
        vector_index.test_vector_search("water", 5)
        
        # Close connection
        vector_index.close()
        
        logger.info("Vector index creation and testing complete")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()