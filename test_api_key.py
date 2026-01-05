import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Print API key (we'll check it)
api_key = config['api']['openweather_key']

print(f"API Key loaded: {api_key}")
print(f"Length: {len(api_key)}")
print(f"Has spaces? {' ' in api_key}")
print(f"First 5 chars: {api_key[:5]}")
print(f"Last 5 chars: {api_key[-5:]}")

