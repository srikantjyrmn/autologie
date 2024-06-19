import ast
import os
import re
import json
import logging
import datetime
import xml.etree.ElementTree as ET
from typing import Dict, Any, List




def extract_json_from_markdown(text):
    """
    Extracts the JSON string from the given text using a regular expression pattern.
    
    Args:
        text (str): The input text containing the JSON string.
        
    Returns:
        dict: The JSON data loaded from the extracted string, or None if the JSON string is not found.
    """
    json_pattern = r'```json\r?\n(.*?)\r?\n```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        json_string = match.group(1)
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON string: {e}")
    else:
        print("JSON string not found in the text.")
    return None


def clean_keys(self, data) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Clean the keys of a Dict"""
    if isinstance(data, dict):
        # Create a new dictionary with modified keys
        new_dict = {}
        for key, value in data.items():
            # Remove the leading 'XXX_' from keys
            new_key = re.sub(r"^\d{3}_", "", key)
            # Recursively clean nested dictionaries and lists
            new_dict[new_key] = self.clean_keys(value)
        return new_dict
    elif isinstance(data, list):
        # Process each item in the list
        return [self.clean_keys(item) for item in data]
    else:
        # Return the item as is if it's not a dict or list
        return data
