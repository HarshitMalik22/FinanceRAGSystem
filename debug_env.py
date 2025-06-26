import os
import sys
from dotenv import load_dotenv

print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())
print("Environment files in current directory:", [f for f in os.listdir('.') if f.startswith('.env')])

# Try to load .env from current directory
env_path = os.path.join(os.getcwd(), '.env')
print(f"\nLoading .env from: {env_path}")
load_dotenv(env_path, override=True)

# Get the API key
api_key = os.getenv("GROQ_API_KEY")
print(f"\nAPI Key from .env: {repr(api_key)}")
print(f"Key length: {len(api_key) if api_key else 0}")
print(f"Key starts with: {api_key[:8] if api_key else 'N/A'}")

# Print all environment variables that might be relevant
print("\nEnvironment variables:")
for k, v in sorted(os.environ.items()):
    if any(x in k.lower() for x in ['key', 'token', 'secret', 'api']):
        print(f"{k} = {v[:8]}... (length: {len(v)})")
