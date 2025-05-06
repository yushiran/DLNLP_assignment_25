"""
Building a Knowledge Graph for Nüshu Characters
This script reads processed CSV data and creates a knowledge graph in Neo4j
"""
import pandas as pd
import os
from neo4j import GraphDatabase
import logging
from utils import translator
from dotenv import load_dotenv
load_dotenv()

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NushuKnowledgeGraph:
    def __init__(self, uri, username, password):
        """
        Initialize Neo4j connection
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            raise

    def close(self):
        """
        Close Neo4j connection
        """
        self.driver.close()
        logger.info("Neo4j connection closed")

    def clear_database(self):
        """
        Clear all nodes and relationships in the database
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database has been cleared")

    def create_constraints(self):
        """
        Create necessary constraints and indexes
        """
        with self.driver.session() as session:
            # Create unique constraint for Nüshu characters
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:NushuCharacter) REQUIRE c.character IS UNIQUE")
            
            # Create unique constraint for Chinese characters
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:ChineseCharacter) REQUIRE c.character IS UNIQUE")
            
            # Create unique constraint for pronunciations
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pronunciation) REQUIRE p.value IS UNIQUE")
            
            # Create unique constraint for English translations
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:EnglishTranslation) REQUIRE e.text IS UNIQUE")
            
            logger.info("Constraints and indexes created")

    def create_nushu_character_node(self, tx, character, sequence):
        """
        Create Nüshu character node
        """
        query = (
            "MERGE (c:NushuCharacter {character: $character}) "
            "ON CREATE SET c.sequence = $sequence "
            "RETURN c"
        )
        result = tx.run(query, character=character, sequence=sequence)
        return result.single()

    def create_chinese_character_node(self, tx, character):
        """
        Create Chinese character node
        """
        query = (
            "MERGE (c:ChineseCharacter {character: $character}) "
            "RETURN c"
        )
        result = tx.run(query, character=character)
        return result.single()

    def create_pronunciation_node(self, tx, pronunciation):
        """
        Create pronunciation node
        """
        query = (
            "MERGE (p:Pronunciation {value: $pronunciation}) "
            "RETURN p"
        )
        result = tx.run(query, pronunciation=pronunciation)
        return result.single()

    def create_english_translation_node(self, tx, english_text):
        """
        Create English translation node
        """
        query = (
            "MERGE (e:EnglishTranslation {text: $english_text}) "
            "RETURN e"
        )
        result = tx.run(query, english_text=english_text)
        return result.single()

    def create_relationships(self, tx, nushu_char, chinese_chars, pronunciation, font_path, english_translation):
        """
        Create relationships between nodes
        """
        # Relationship between Nüshu character and Chinese characters
        for chinese_char in chinese_chars:
            if chinese_char.strip():  # Ensure the character is not blank
                query = (
                    "MATCH (n:NushuCharacter {character: $nushu_char}) "
                    "MATCH (c:ChineseCharacter {character: $chinese_char}) "
                    "MERGE (n)-[r:CORRESPONDS_TO]->(c) "
                    "RETURN r"
                )
                tx.run(query, nushu_char=nushu_char, chinese_char=chinese_char)

                # Create relationship between Chinese character and English translation
                query = (
                    "MATCH (c:ChineseCharacter {character: $chinese_char}) "
                    "MATCH (e:EnglishTranslation {text: $english_text}) "
                    "MERGE (c)-[r:TRANSLATES_TO]->(e) "
                    "RETURN r"
                )
                tx.run(query, chinese_char=chinese_char, english_text=english_translation)
             
        # Relationship between Nüshu character and pronunciation
        query = (
            "MATCH (n:NushuCharacter {character: $nushu_char}) "
            "MATCH (p:Pronunciation {value: $pronunciation}) "
            "MERGE (n)-[r:PRONOUNCED_AS]->(p) "
            "RETURN r"
        )
        tx.run(query, nushu_char=nushu_char, pronunciation=pronunciation)
        
        # Set the font path for Nüshu character
        query = (
            "MATCH (n:NushuCharacter {character: $nushu_char}) "
            "SET n.font_path = $font_path "
            "RETURN n"
        )
        tx.run(query, nushu_char=nushu_char, font_path=font_path)

    def build_knowledge_graph(self, csv_path):
        """
        Build knowledge graph from CSV file
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully read CSV file: {csv_path}")
            
            with self.driver.session() as session:
                # Clear database and create constraints
                self.clear_database()
                self.create_constraints()
                
                # Iterate through data and create nodes and relationships
                total_rows = len(df)
                for index, row in df.iterrows():
                    nushu_char = row['FL Character']
                    sequence = row['《Zitie》Sequence']
                    chinese_chars = row['Corresponding Chinese Character'].split()
                    
                    pronunciation = row['Jiangyong Dialect Pronunciation']
                    font_path = row['font_path']
                    
                    session.execute_write(self.create_nushu_character_node, nushu_char, sequence)
                    session.execute_write(self.create_pronunciation_node, pronunciation)
                    for chinese_char in chinese_chars[0]:
                        session.execute_write(self.create_chinese_character_node, chinese_char)
                        
                        english_translation = translator.translate_chinese_to_english(chinese_char)
                        if english_translation:
                            session.execute_write(self.create_english_translation_node, english_translation)

                        session.execute_write(self.create_relationships, nushu_char, chinese_char, pronunciation, font_path, english_translation)

                    if index % 50 == 0:
                        logger.info(f"Processed {index+1}/{total_rows} records")
                
                logger.info(f"Knowledge graph construction complete, processed {total_rows} records")
                
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise

def main():
    try:
        # First try to read from environment variables
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USERNAME', 'neo4j')
        ne4j_pwd = os.environ.get('NEO4J_PASSWORD')
        
        # Get CSV file path
        data_dir = os.path.abspath(os.path.join(base_dir, 'data', 'processed'))
        csv_path = os.path.join(data_dir, 'data.csv')
        
        # Create knowledge graph
        kg = NushuKnowledgeGraph(neo4j_uri, neo4j_user, ne4j_pwd)
        kg.build_knowledge_graph(csv_path)
        kg.close()
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()