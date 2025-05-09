#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pathlib import Path
# Get the absolute path to the project root directory (two levels up from the current file)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def add_font_paths_to_csv():
    """
    Read data.csv, find matching images in nushu_font folder, 
    add a new column with image paths, and save the result to processed directory.
    """
    # Define file paths
    data_path = os.path.join(base_dir,'data/raw/data.csv')
    font_dir = os.path.join(base_dir,'data/raw/nushu_font')
    processed_dir = os.path.join(base_dir,'data/processed')
    output_path = os.path.join(processed_dir,'data.csv')
    
    # Create processed directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading CSV from {data_path}")
    df = pd.read_csv(Path(data_path))
    
    # Get list of all available font image filenames
    font_images = os.listdir(font_dir)
    
    # Create a new column for image paths
    image_paths = []
    
    # For each character in the first column, find the matching image
    for char in df['FL Character']:
        # Image filename would be the character with .jpg extension
        image_filename = f"{char}.jpg"
        
        if image_filename in font_images:
            # If found, add the path to the list 
            image_path = str('https://github.com/yushiran/DLNLP_assignment_25/tree/main/data/raw/nushu_font/' + image_filename)
            image_paths.append(image_path)
        else:
            # If not found, add empty string
            image_paths.append("")
            print(f"Warning: No image found for character '{char}'")
    
    # Add the image paths as a new column
    df['font_path'] = image_paths
    
    # Save the updated dataframe to the processed directory
    print(f"Saving processed CSV to {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"Done! Added {len([p for p in image_paths if p])} image paths out of {len(df)} characters.")


if __name__ == "__main__":
    add_font_paths_to_csv()