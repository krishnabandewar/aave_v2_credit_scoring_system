import requests
import json
import zipfile
import tempfile
import os
from typing import List, Dict, Any, Optional

def download_file_from_google_drive(file_id: str, file_type: str = "json") -> Optional[List[Dict[Any, Any]]]:
    """
    Download and process file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        file_type: Type of file ("json" or "zip")
        
    Returns:
        List of transaction dictionaries or None if failed
    """
    try:
        # Google Drive download URL
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Create a session for handling redirects
        session = requests.Session()
        
        # First request to get the download warning page
        response = session.get(url, stream=True)
        
        # Look for the download confirmation token
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        
        # If we found a token, make the confirmed download request
        if token:
            params = {'confirm': token}
            response = session.get(url, params=params, stream=True)
        
        # Check if request was successful
        if response.status_code != 200:
            print(f"Failed to download file. Status code: {response.status_code}")
            return None
        
        # Process based on file type
        if file_type.lower() == "json":
            return _process_json_response(response)
        elif file_type.lower() == "zip":
            return _process_zip_response(response)
        else:
            print(f"Unsupported file type: {file_type}")
            return None
            
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None

def _process_json_response(response) -> Optional[List[Dict[Any, Any]]]:
    """Process JSON response from Google Drive"""
    try:
        # Try to load JSON directly
        content = response.content.decode('utf-8')
        data = json.loads(content)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If it's a dict, try to find the list of transactions
            for key, value in data.items():
                if isinstance(value, list):
                    return value
            return [data]  # Return as single-item list
        else:
            print("Unexpected JSON structure")
            return None
            
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing JSON response: {str(e)}")
        return None

def _process_zip_response(response) -> Optional[List[Dict[Any, Any]]]:
    """Process ZIP response from Google Drive"""
    try:
        # Save response content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_zip_path = temp_file.name
        
        # Extract and process the ZIP file
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            # List files in the ZIP
            file_list = zip_ref.namelist()
            
            # Look for JSON files
            json_files = [f for f in file_list if f.endswith('.json')]
            
            if not json_files:
                print("No JSON files found in ZIP archive")
                return None
            
            # Process the first JSON file found
            json_file = json_files[0]
            
            with zip_ref.open(json_file) as json_file_handle:
                content = json_file_handle.read().decode('utf-8')
                data = json.loads(content)
                
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # If it's a dict, try to find the list of transactions
                    for key, value in data.items():
                        if isinstance(value, list):
                            return value
                    return [data]  # Return as single-item list
        
        # Clean up temporary file
        os.unlink(temp_zip_path)
        
    except zipfile.BadZipFile:
        print("Downloaded file is not a valid ZIP archive")
        return None
    except Exception as e:
        print(f"Error processing ZIP response: {str(e)}")
        return None
    
    return None

def validate_transaction_data(data: List[Dict[Any, Any]]) -> bool:
    """
    Validate that the transaction data has the expected structure
    
    Args:
        data: List of transaction dictionaries
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if not isinstance(data, list):
        print("Data is not a list")
        return False
    
    if len(data) == 0:
        print("Data list is empty")
        return False
    
    # Check first few transactions for required fields
    required_fields = ['user', 'action', 'amount']
    
    # Print sample transaction structure for debugging
    if len(data) > 0:
        print("Sample transaction structure:")
        sample_tx = data[0]
        print(f"Keys: {list(sample_tx.keys()) if isinstance(sample_tx, dict) else 'Not a dict'}")
        print(f"Sample: {sample_tx}")
    
    for i, transaction in enumerate(data[:10]):  # Check first 10 transactions
        if not isinstance(transaction, dict):
            print(f"Transaction {i} is not a dictionary")
            return False
        
        # Check for common alternative field names
        user_field = None
        amount_field = None
        action_field = None
        
        for key in transaction.keys():
            key_lower = key.lower()
            if 'user' in key_lower or 'address' in key_lower or 'wallet' in key_lower:
                user_field = key
            elif 'amount' in key_lower or 'value' in key_lower:
                amount_field = key
            elif 'action' in key_lower or 'type' in key_lower or 'event' in key_lower:
                action_field = key
        
        # Check if we found the essential fields (with flexible naming)
        missing_fields = []
        if not user_field:
            missing_fields.append('user/address/wallet')
        if not amount_field:
            missing_fields.append('amount/value')
        if not action_field:
            missing_fields.append('action/type/event')
        
        if missing_fields:
            print(f"Transaction {i} missing required fields: {missing_fields}")
            print(f"Available fields: {list(transaction.keys())}")
            return False
    
    print(f"Data validation passed. Found {len(data)} transactions.")
    return True

def format_large_number(number: float) -> str:
    """Format large numbers for display"""
    if number >= 1e9:
        return f"{number/1e9:.1f}B"
    elif number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.2f}"

def calculate_percentile_ranges(scores: List[float], num_ranges: int = 10) -> List[tuple]:
    """Calculate percentile-based score ranges"""
    import numpy as np
    
    percentiles = np.linspace(0, 100, num_ranges + 1)
    ranges = []
    
    for i in range(len(percentiles) - 1):
        min_score = np.percentile(scores, percentiles[i])
        max_score = np.percentile(scores, percentiles[i + 1])
        ranges.append((min_score, max_score))
    
    return ranges
