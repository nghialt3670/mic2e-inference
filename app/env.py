import os

from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model loading flags
LOAD_SAM3 = os.getenv("LOAD_SAM3", "false").lower() == "true"
LOAD_OBJECT_CLEAR = os.getenv("LOAD_OBJECT_CLEAR", "false").lower() == "true"
LOAD_BOX_DIFF = os.getenv("LOAD_BOX_DIFF", "false").lower() == "true"
LOAD_GLIGEN = os.getenv("LOAD_GLIGEN", "true").lower() == "true"
LOAD_FLUX = os.getenv("LOAD_FLUX", "false").lower() == "true"
