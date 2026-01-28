"""Launch the RLM Multi-Turn Chat UI.

This is the main entry point for the Irys RLM user interface.
Uses the multi-turn chat interface for conversational document analysis.
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[OK] Loaded environment variables from {env_path}")
    else:
        print(f"[WARN] No .env file found at {env_path}")
        print("   You can create one from .env.example")
except ImportError:
    print("[WARN] python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Falling back to system environment variables...")

# Now import after env is loaded - use multi-turn chat UI
from irys.ui.chat_app import create_chat_app


if __name__ == "__main__":
    # Check for API key in environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY environment variable not set!")
        print("\nOptions:")
        print("  1. Create a .env file with: GEMINI_API_KEY=your-key-here")
        print("  2. Or export it: export GEMINI_API_KEY='your-key-here'")
        print("  3. Or copy .env.example to .env and edit it")
        sys.exit(1)

    print(f"[OK] API key loaded: {api_key[:20]}...")

    app = create_chat_app(api_key=api_key)
    port = int(os.environ.get("GRADIO_SERVER_PORT", 7863))
    print(f"\n[START] Launching Irys RLM Multi-Turn Chat UI on http://localhost:{port}")
    app.launch(server_port=port)
