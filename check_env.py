import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

# Get the API key
api_key = os.getenv("GROQ_API_KEY")

print(f"API Key: {repr(api_key)}")
print(f"Length: {len(api_key) if api_key else 0} characters")
print(f"Is printable: {api_key and all(32 <= ord(c) <= 126 for c in api_key)}")
print(f"Stripped length: {len(api_key.strip()) if api_key else 0} characters")
