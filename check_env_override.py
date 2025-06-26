import os
import sys
from dotenv import load_dotenv

def print_env_var(var_name):
    value = os.getenv(var_name)
    print(f"{var_name}: {repr(value)} (length: {len(value) if value else 0})")

print("=== Environment Debug ===")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")
print(f"Environment files: {[f for f in os.listdir('.') if f.startswith('.env')]}")

# Try loading .env from current directory
env_path = os.path.join(os.getcwd(), '.env')
print(f"\nLoading .env from: {env_path}")
load_dotenv(env_path, override=True)

# Check GROQ_API_KEY in different ways
print("\n=== GROQ_API_KEY Check ===")
print_env_var("GROQ_API_KEY")

# Check if the key is being overridden somewhere
print("\n=== Environment Variables ===")
for k, v in sorted(os.environ.items()):
    if 'key' in k.lower() or 'token' in k.lower() or 'secret' in k.lower() or 'api' in k.lower():
        print(f"{k} = {v[:4]}... (length: {len(v)})")

# Try to read the .env file directly
print("\n=== .env File Content ===")
try:
    with open(env_path, 'r') as f:
        content = f.read()
    print(f"File exists. First 200 chars:\n{content[:200]}")
    print(f"\nFile length: {len(content)} bytes")
    print(f"Contains GROQ_API_KEY: {'GROQ_API_KEY' in content}")
except Exception as e:
    print(f"Error reading .env file: {e}")
