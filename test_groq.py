import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("GROQ_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"Key length: {len(api_key) if api_key else 0} characters")
print(f"Key starts with: {api_key[:10] if api_key else 'N/A'}..." if api_key else "No key")

# Try to use the key
try:
    from groq import Groq
    client = Groq(api_key=api_key)
    print("\nTesting API key with Groq client...")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}],
        model="llama-3.3-70b-versatile",
    )
    print("\nAPI Key is valid! Response:")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print(f"\nError using API key: {str(e)}")
    if hasattr(e, 'response') and hasattr(e.response, 'text'):
        print(f"Response: {e.response.text}")
