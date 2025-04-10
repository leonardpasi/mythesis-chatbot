import hashlib
import json
import os


def get_config_hash(config: dict) -> str:
    # Use JSON to serialize and sort keys for deterministic output
    config_str = json.dumps(config, sort_keys=True)

    return hashlib.sha256(config_str.encode()).hexdigest()[:10]  # short hash


def get_openai_api_key():
    """
    Get the OpenAI API key from an environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    raise ValueError(
        "OpenAI API key not found. Please follow the instruction in the readme file."
    )
