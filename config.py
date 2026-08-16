import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR="data"
OUTPUT_DATA_DIR="output_data"

# Параметры чанкования
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100