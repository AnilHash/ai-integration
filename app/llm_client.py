from langfuse.openai import openai  # type: ignore
import os

OLLAMA_BASE_URL = "http://localhost:11434/v1"

client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
