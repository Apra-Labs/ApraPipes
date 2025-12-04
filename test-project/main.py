"""
Simple test module for autonomous agent testing

FIXED: Added json import
"""
from datetime import datetime
import json

def calculate_age(birth_year):
    """Calculate age from birth year"""
    current_year = datetime.now().year
    age = current_year - birth_year
    return age

def format_greeting(name, birth_year):
    """Format a greeting with age"""
    age = calculate_age(birth_year)

    # Fixed: json module now imported
    data = json.dumps({
        "greeting": f"Hello {name}!",
        "age": age,
        "message": f"You are {age} years old"
    })
    return data

if __name__ == "__main__":
    result = format_greeting("Test User", 1990)
    print(result)
