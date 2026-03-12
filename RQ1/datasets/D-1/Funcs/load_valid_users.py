#!/usr/bin/env python3
"""
Module to load valid users for Preprocessing_hourly.ipynb
"""
import os

def load_valid_users(file_path='valid_users.txt'):
    """
    Load the list of valid users from the valid_users.txt file.
    
    Args:
        file_path (str): Path to the valid_users.txt file
        
    Returns:
        list: List of valid user IDs
    """
    valid_users = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    valid_users.append(line)
        
        print(f"✅ Loaded {len(valid_users)} valid users")
        return valid_users
        
    except FileNotFoundError:
        print(f"❌ Error: {file_path} not found")
        return []
    except Exception as e:
        print(f"❌ Error loading valid users: {e}")
        return []

def get_problematic_users():
    """
    Return the list of problematic users that should be avoided.
    
    Returns:
        list: List of problematic user IDs
    """
    return [
        'P035', 'P036', 'P041', 'P042', 'P049', 'P053', 'P070', 'P073', 
        'P076', 'P080', 'P082', 'P096', 'P097', 'P113', 'P119', 'P128', 'P137'
    ]

def filter_users_by_validity(user_list, valid_users=None):
    """
    Filter a list of users to only include valid ones.
    
    Args:
        user_list (list): List of user IDs to filter
        valid_users (list, optional): List of valid users. If None, loads from file.
        
    Returns:
        list: Filtered list of valid users
    """
    if valid_users is None:
        valid_users = load_valid_users()
    
    filtered_users = [user for user in user_list if user in valid_users]
    print(f"📊 Filtered {len(user_list)} users to {len(filtered_users)} valid users")
    
    return filtered_users

# Example usage:
if __name__ == "__main__":
    valid_users = load_valid_users()
    print(f"Valid users: {valid_users[:5]}...")  # Show first 5
    
    problematic_users = get_problematic_users()
    print(f"Problematic users: {problematic_users}") 
