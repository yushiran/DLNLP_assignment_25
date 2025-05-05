import deepl
import logging
import os
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set DeepL library's logger to only show warnings and errors
logging.getLogger("deepl").setLevel(logging.WARNING)

def translate_chinese_to_english(chinese_char):
    """
    Translate Chinese character to English using DeepL API
    """
    auth_key = os.environ.get('DEEPL_API_KEY') # Replace with your key
    deepl_client = deepl.Translator(auth_key)
    try:
        result = deepl_client.translate_text(chinese_char, target_lang="EN-GB", source_lang="ZH")
        return result.text
    except Exception as e:
        logger.error(f"Error translating {chinese_char}: {e}")
        return None