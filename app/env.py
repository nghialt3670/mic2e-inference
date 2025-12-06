import os

from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
