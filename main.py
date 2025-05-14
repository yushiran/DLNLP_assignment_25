from src import create_knowledge_graph, add_vector_index_to_kg, deploy_rag_with_deepseek
# from src.utils import translator
# from src.utils import add_font_toCSV
import pandas as pd
import os
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # process CSV data
    # add_font_toCSV.add_font_paths_to_csv()
    
    # create knowledge graph
    # create_knowledge_graph.main()

    # add vector index to the knowledge graph
    # add_vector_index_to_kg.main()
    
    # Run the Streamlit interface
    deploy_rag_with_deepseek.run_rag_interface()

    return 0

if __name__ == "__main__":
    main()