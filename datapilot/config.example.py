"""
DataPilot Configuration
Copy this file to config.py and add your API keys.

Your data stays local - only queries are sent to the AI provider.
"""

# Choose your AI provider: 'openai' or 'anthropic'
PROVIDER = 'openai'

# API Keys (get yours at https://platform.openai.com or https://console.anthropic.com)
OPENAI_API_KEY = 'your-openai-api-key-here'
ANTHROPIC_API_KEY = 'your-anthropic-api-key-here'

# Optional: Specify model (leave empty for defaults)
# OpenAI: 'gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo'
# Anthropic: 'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'
MODEL = ''
