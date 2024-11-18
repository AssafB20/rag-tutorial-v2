import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


# Load environment variables from .env file
load_dotenv()


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
        
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",  # This is OpenAI's latest and most efficient embedding model
        openai_api_key=api_key
    )
    return embeddings
