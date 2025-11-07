import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main_gemini import app

# Export the FastAPI app for Vercel
handler = app
