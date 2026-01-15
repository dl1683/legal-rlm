"""Launch the RLM UI."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set API key
os.environ["GEMINI_API_KEY"] = "AIzaSyCJ0_zdC9mA6Jkh4F98UFwm9AuwXp3wICg"

from irys.ui.app import create_app

if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7862)
