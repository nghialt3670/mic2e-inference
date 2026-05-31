import os

from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT"))
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model loading flags
LOAD_SAM3 = os.getenv("LOAD_SAM3", "false").lower() == "true"
LOAD_OBJECT_CLEAR = os.getenv("LOAD_OBJECT_CLEAR", "false").lower() == "true"
LOAD_BOX_DIFF = os.getenv("LOAD_BOX_DIFF", "false").lower() == "true"
LOAD_GLIGEN = os.getenv("LOAD_GLIGEN", "false").lower() == "true"
LOAD_SD_INPAINT = os.getenv("LOAD_SD_INPAINT", "false").lower() == "true"
LOAD_FLUX = os.getenv("LOAD_FLUX", "false").lower() == "true"

# GLIGEN checkpoint configuration
GLIGEN_AUTO_DOWNLOAD_GENERATION = os.getenv("GLIGEN_AUTO_DOWNLOAD_GENERATION", "false").lower() == "true"
GLIGEN_AUTO_DOWNLOAD_INPAINTING = os.getenv("GLIGEN_AUTO_DOWNLOAD_INPAINTING", "false").lower() == "true"
GLIGEN_CHECKPOINT_DIR = os.getenv("GLIGEN_CHECKPOINT_DIR", "")

# Aesthetic Regressor configuration
AESTHETIC_REGRESSOR_WEIGHT_PATH = os.getenv("AESTHETIC_REGRESSOR_WEIGHT_PATH", "app/external/AestheticEnhancer/model/epoch_25.pth")
AESTHETIC_REGRESSOR_BACKBONE = os.getenv("AESTHETIC_REGRESSOR_BACKBONE", "resnet18")
AESTHETIC_REGRESSOR_N_FACTORS = int(os.getenv("AESTHETIC_REGRESSOR_N_FACTORS", "5")) if os.getenv("AESTHETIC_REGRESSOR_N_FACTORS") else None
AESTHETIC_REGRESSOR_ACTIVATION = os.getenv("AESTHETIC_REGRESSOR_ACTIVATION", "tanh")
