import os


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
