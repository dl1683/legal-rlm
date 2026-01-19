#!/usr/bin/env python3
"""Simple test script for Irys RLM system."""

import asyncio
import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed (optional)")
    pass


async def test_basic_investigation():
    """Test basic investigation functionality."""
    print("=" * 70)
    print("IRYS RLM - BASIC INVESTIGATION TEST")
    print("=" * 70)

    # Import Irys
    try:
        from irys import Irys
        print("✓ Successfully imported Irys")
    except ImportError as e:
        print(f"✗ Failed to import Irys: {e}")
        return

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n✗ ERROR: GEMINI_API_KEY not set in environment")
        print("   Please set it with: export GEMINI_API_KEY='your-key-here'")
        return

    print(f"✓ API key found: {api_key[:20]}...")

    # Check test documents
    test_docs = Path("test_documents")
    if not test_docs.exists():
        print(f"\n✗ Test documents directory not found: {test_docs}")
        return

    files = list(test_docs.glob("*.txt"))
    print(f"✓ Found {len(files)} test documents:")
    for f in files:
        print(f"  - {f.name}")

    # Initialize Irys
    print("\n" + "=" * 70)
    print("INITIALIZING IRYS...")
    print("=" * 70)

    try:
        irys = Irys(api_key=api_key)
        print("✓ Irys initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Irys: {e}")
        return

    # Run investigation
    print("\n" + "=" * 70)
    print("RUNNING INVESTIGATION...")
    print("=" * 70)

    query = "What are the key obligations and payment terms in the contract?"
    print(f"\nQuery: {query}")
    print(f"Repository: {test_docs}")
    print("\nThis may take 30-60 seconds...\n")

    try:
        result = await irys.investigate(
            query=query,
            repository=str(test_docs)
        )

        print("\n" + "=" * 70)
        print("INVESTIGATION COMPLETE!")
        print("=" * 70)

        print(f"\n📊 STATISTICS:")
        print(f"  - Documents read: {result.state.documents_read}")
        print(f"  - Searches performed: {result.state.searches_performed}")
        print(f"  - Citations found: {len(result.state.citations)}")
        print(f"  - Entities extracted: {len(result.state.entities)}")
        print(f"  - Thinking steps: {len(result.state.thinking_steps)}")
        print(f"  - Confidence: {result.confidence}%")

        print(f"\n📝 OUTPUT:")
        print("-" * 70)
        print(result.output)
        print("-" * 70)

        if result.state.citations:
            print(f"\n📚 CITATIONS ({len(result.state.citations)}):")
            for i, citation in enumerate(result.state.citations[:5], 1):
                print(f"\n  {i}. {citation.text[:100]}...")
                print(f"     Source: {citation.source}")
                if citation.page:
                    print(f"     Page: {citation.page}")

        if result.state.entities:
            print(f"\n👥 ENTITIES FOUND:")
            entity_types = {}
            for entity in result.state.entities:
                entity_types[entity.type] = entity_types.get(
                    entity.type, 0) + 1
            for etype, count in entity_types.items():
                print(f"  - {etype}: {count}")

        print("\n" + "=" * 70)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Investigation failed: {e}")
        import traceback
        traceback.print_exc()


async def test_quick_search():
    """Test quick search functionality."""
    print("\n" + "=" * 70)
    print("TESTING QUICK SEARCH")
    print("=" * 70)

    try:
        from irys import quick_search

        results = await quick_search("payment", "./test_documents")

        print(f"\n✓ Found {len(results)} matches for 'payment'")

        if results:
            print(f"\n  First match:")
            hit = results[0]
            print(f"  - File: {hit['file']}")
            print(f"  - Page: {hit.get('page', 'N/A')}")
            print(f"  - Text: {hit['text'][:100]}...")

    except Exception as e:
        print(f"✗ Quick search failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n🚀 Starting Irys RLM Tests...\n")

    # Run async tests
    asyncio.run(test_basic_investigation())
    asyncio.run(test_quick_search())

    print("\n✅ All tests completed!\n")


if __name__ == "__main__":
    main()
