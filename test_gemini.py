import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any

# Load environment variables
load_dotenv(override=True)

def get_required_env_var(var_name: str) -> str:
    """Get required environment variable or exit if not found."""
    value = os.getenv(var_name)
    if not value:
        print(f"Error: {var_name} not found in environment variables")
        print(f"Please add {var_name} to your .env file")
        sys.exit(1)
    return value

def list_available_models(api_key: str) -> List[Dict[str, Any]]:
    """List all available Gemini models."""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [{
            'name': model.name,
            'display_name': getattr(model, 'display_name', 'N/A'),
            'description': getattr(model, 'description', 'N/A'),
            'methods': list(getattr(model, 'supported_generation_methods', [])),
        } for model in models]
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return []

def test_gemini_generation(api_key: str, model_name: str = "gemini-1.5-pro") -> Optional[str]:
    """Test text generation with the specified Gemini model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Test with a simple prompt
        response = model.generate_content("Hello, Gemini! Can you tell me a short fact about AI?")
        
        if not response.text:
            print(f"No response text received from model {model_name}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print(f"Prompt feedback: {response.prompt_feedback}")
            return None
            
        return response.text
        
    except Exception as e:
        print(f"Error during generation with model {model_name}: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def main():
    # Get API key from environment
    api_key = get_required_env_var('GOOGLE_API_KEY')
    model_name = os.getenv('GOOGLE_MODEL', 'gemini-1.5-pro')
    
    print(f"Using model: {model_name}")
    
    # List available models
    print("\nFetching available models...")
    models = list_available_models(api_key)
    
    if not models:
        print("No models found. Please check your API key and network connection.")
        return
    
    print(f"\nFound {len(models)} available models:")
    for i, model in enumerate(models[:5], 1):  # Show first 5 models
        print(f"{i}. {model['name']} (Methods: {', '.join(model['methods'])})")
    
    # Test generation with specified model
    print(f"\nTesting generation with model: {model_name}")
    response = test_gemini_generation(api_key, model_name)
    
    if response:
        print("\nGeneration successful!")
        print(f"Response: {response}")
    else:
        print("\nGeneration test failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
