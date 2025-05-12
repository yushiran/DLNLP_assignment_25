"""
Unicode helper functions for proper rendering of Nüshu characters
This module provides utility functions to ensure proper handling of Unicode characters,
particularly for the special Nüshu script characters.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Add a dedicated function to detect and extract Nüshu characters
def extract_nushu_characters(text):
    """
    Extract Nüshu characters from text and return them with their Unicode codepoints
    
    Args:
        text: The text to process
        
    Returns:
        List of tuples with (character, codepoint)
    """
    if not isinstance(text, str):
        return []
    
    try:
        # Find all Nüshu characters in the text
        nushu_chars = re.findall(r'[\U0001B170-\U0001B2FF]', text)
        
        # Return character and codepoint information
        result = []
        for char in nushu_chars:
            codepoint = hex(ord(char)).upper()
            result.append((char, codepoint))
        
        return result
    except Exception as e:
        logger.error(f"Error extracting Nüshu characters: {e}")
        return []

def ensure_proper_encoding(text):
    """
    Ensure text is properly encoded in UTF-8 with proper handling of special characters
    
    Args:
        text: The text to process
        
    Returns:
        Properly encoded text
    """
    if not isinstance(text, str):
        return text
        
    try:
        # Normalize Unicode characters (important for handling combining characters)
        # Use NFC normalization for better compatibility
        text = unicodedata.normalize('NFC', text)
        
        # Handle encoding issues more carefully
        # First encode as UTF-8, then decode, but handle errors more gracefully
        text_bytes = text.encode('utf-8', errors='surrogateescape')
        text = text_bytes.decode('utf-8', errors='replace')
        
        # Special handling for Nüshu Unicode block (U+1B170 to U+1B2FF)
        # Replace any placeholder text with actual characters when possible
        text = text.replace('女书字符', '𛅰')  # Generic replacement
        
        # Check for common Nüshu character patterns and ensure they are properly encoded
        nushu_pattern = re.compile(r'(𛅰|𛅱|𛅲|𛅳|𛅴|𛅵|𛅶|𛅷|𛅸|𛅹|𛅺|𛅻|𛅼|𛅽|𛅾|𛅿|𛆀|𛆁|𛆂|𛆃|𛆄)')
        
        # Replace any broken representations with proper Unicode
        for match in nushu_pattern.finditer(text):
            char = match.group(0)
            codepoint = ord(char)
            # Ensure it's properly encoded by regenerating from codepoint
            correct_char = chr(codepoint)
            if char != correct_char:
                text = text.replace(char, correct_char)
        
        return text
    except Exception as e:
        logger.error(f"Error ensuring proper encoding: {e}")
        # Return original text if we encounter errors during processing
        return text

def remove_repetitive_content(text):
    """
    Remove repetitive sections from text to avoid the common model issue
    of repeating the same content multiple times
    
    Args:
        text: The text to process
        
    Returns:
        Text with repetitive sections removed
    """
    if not isinstance(text, str):
        return text
        
    try:
        # Split text into lines
        lines = text.split('\n')
        unique_lines = []
        seen_patterns = set()
        
        for line in lines:
            # Create a simplified fingerprint for comparison (to detect repetitions)
            # Remove spaces, lowercase, and keep only alphanumeric chars for comparison
            simplified = ''.join(c.lower() for c in line if c.isalnum())
            
            # Skip empty lines or those that are just spaces
            if not simplified:
                unique_lines.append(line)  # Keep empty lines for formatting
                continue
                
            # If we haven't seen this pattern before, add it
            if simplified not in seen_patterns:
                seen_patterns.add(simplified)
                unique_lines.append(line)
        
        # Reconstruct text
        result = '\n'.join(unique_lines)
        return result
    except Exception as e:
        logger.error(f"Error removing repetitive content: {e}")
        return text

def clean_model_output(text):
    """
    Clean the model output by fixing common formatting issues
    
    Args:
        text: The text to process
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return text
        
    try:
        # Remove common formatting tags 
        text = re.sub(r'<answer>|</answer>|<response>|</response>', '', text)
        
        # Extract content after system tags if present
        if "</system>" in text:
            text = text.split("</system>", 1)[-1].strip()
        if "<question>" in text and "</question>" in text:
            question_part = text.split("<question>", 1)[1].split("</question>", 1)[0].strip()
            answer_part = text.split("</question>", 1)[-1].strip()
            # Keep only the answer part
            text = answer_part
            
        # Remove common artifacts from model outputs
        text = re.sub(r'<.*?>', '', text)  # Remove any remaining XML/HTML-like tags
        
        # Fix potential Unicode escape sequences
        text = text.replace('\\u', '\\\\u')
        
        # Remove extra asterisks which are common in repeated content
        text = re.sub(r'\*{3,}', '***', text)
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning model output: {e}")
        return text

def process_nushu_text(text):
    """
    Process text that contains Nüshu characters for proper display
    
    Args:
        text: The text containing Nüshu characters
        
    Returns:
        Processed text with proper Nüshu character rendering
    """
    # First ensure proper encoding
    text = ensure_proper_encoding(text)
    
    # Extra processing for Nüshu characters (U+1B170 to U+1B2FF)
    # Replace any potential broken sequences with their proper Unicode
    for i in range(0x1B170, 0x1B300):
        char = chr(i)
        # Check for broken representations of this character
        broken = char.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        if broken != char:
            text = text.replace(broken, char)
    
    # Apply additional processing
    text = remove_repetitive_content(text)
    text = clean_model_output(text)
    
    # Make sure actual Nüshu characters are preserved
    nushu_regex = r'[\U0001B170-\U0001B2FF]'
    matches = re.findall(nushu_regex, text)
    for match in matches:
        # Ensure each Nüshu character is properly represented
        # Replace with the same character but directly as Unicode literal
        codepoint = ord(match)
        safe_char = chr(codepoint)
        # Only replace if needed
        if match != safe_char:
            text = text.replace(match, safe_char)
    
    return text
