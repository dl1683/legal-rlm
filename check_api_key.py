#!/usr/bin/env python3
"""Check API key configuration."""

import os
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Found .env file at: {env_path}")
    else:
        print(f"✗ No .env file found at: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed")

# Check API key
api_key = os.environ.get("GEMINI_API_KEY")
print(f"api_key: {api_key}")

print("\n" + "=" * 70)
print("API KEY DIAGNOSTICS")
print("=" * 70)

if not api_key:
    print("\n✗ GEMINI_API_KEY is not set!")
    print("\nTroubleshooting:")
    print("1. Check if .env file exists")
    print("2. Make sure it contains: GEMINI_API_KEY=your_key_here")
    print("3. No quotes, no spaces around =")
else:
    print(f"\n✓ API key found!")
    print(f"  Length: {len(api_key)} characters")
    print(f"  First 20 chars: {api_key[:20]}...")
    print(f"  Last 10 chars: ...{api_key[-10:]}")

    # Check for common issues
    issues = []

    if api_key.startswith('"') or api_key.startswith("'"):
        issues.append(
            "⚠️  Key starts with a quote - remove quotes from .env file")

    if api_key.endswith('"') or api_key.endswith("'"):
        issues.append(
            "⚠️  Key ends with a quote - remove quotes from .env file")

    if ' ' in api_key:
        issues.append("⚠️  Key contains spaces - remove spaces")

    if '\n' in api_key or '\r' in api_key:
        issues.append("⚠️  Key contains newlines - check .env file format")

    # Check expected format
    if not api_key.startswith("AIzaSy"):
        issues.append("⚠️  Key doesn't start with 'AIzaSy' - might be invalid")

    if len(api_key) != 39:
        issues.append(
            f"⚠️  Key length is {len(api_key)}, expected 39 characters")

    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nYour .env file should look like:")
        print("  GEMINI_API_KEY=AIzaSyBF_idy6nP53a87...")
        print("  (no quotes, no spaces)")
    else:
        print("\n✅ API key format looks good!")
        print("\nTesting with Google API...")

        try:
            from google import genai
            client = genai.Client(api_key=api_key)

            # Try a simple request
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents='Say "Hello"'
            )

            print("✅ API key is VALID and working!")
            print(f"   Response: {response.text[:50]}...")

        except Exception as e:
            print(f"❌ API key test FAILED: {e}")
            print("\nPossible reasons:")
            print("  1. API key is invalid or revoked")
            print("  2. API key doesn't have proper permissions")
            print("  3. Gemini API is not enabled for this key")
            print("\nGet a new key at: https://aistudio.google.com/app/apikey")

print("\n" + "=" * 70)
