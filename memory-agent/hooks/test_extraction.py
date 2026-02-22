#!/usr/bin/env python3
"""Quick test for the extraction logic."""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_memories import extract_from_text, run_extraction, content_hash

SAMPLE_TRANSCRIPT = (
    "User: Can you fix the login bug?\n"
    "\n"
    "Assistant: I found the root cause of the authentication issue.\n"
    "The bug was in the session handler - it was not checking token expiry correctly.\n"
    "Error: TokenExpiredError raised when refreshing with stale token.\n"
    "\n"
    "I decided to use JWT with sliding window expiry instead of fixed tokens.\n"
    "Let's use Redis for session caching since it supports TTL natively.\n"
    "Going with bcrypt for password hashing, argon2 was too slow for our use case.\n"
    "\n"
    "The pattern for auth middleware is: validate token first, then check permissions,\n"
    "then load user context. This three-step approach prevents unnecessary DB calls.\n"
    "\n"
    "Fixed the issue by adding proper token refresh logic before expiry check.\n"
    "The approach is to always refresh tokens that are within 5 minutes of expiry.\n"
    "\n"
    "Architecture: Use a gateway pattern for all API routes.\n"
    "Convention: All error responses must include an error_code field.\n"
    "\n"
    "Root cause: The session middleware was comparing timestamps without timezone info.\n"
)


def test_extraction():
    """Test that extraction finds decisions, errors, and patterns."""
    results = extract_from_text(SAMPLE_TRANSCRIPT, set())

    print(f"Extracted {len(results)} memories:\n")
    for i, r in enumerate(results):
        print(f"  [{i+1}] type={r['type']:<10} importance={r['importance']} tags={r['tags']}")
        content_preview = r['content'][:120].replace('\n', ' ')
        print(f"       content: {content_preview}...")
        print()

    # Verify we got at least some of each type
    types_found = {r['type'] for r in results}
    print(f"Types found: {types_found}")

    assert "decision" in types_found, "Should have found at least one decision"
    assert "error" in types_found, "Should have found at least one error"
    assert "code" in types_found, "Should have found at least one pattern/code"
    print("\nAll assertions passed!")


def test_dedup():
    """Test that within-run dedup works (no exact duplicate content)."""
    results = extract_from_text(SAMPLE_TRANSCRIPT, set())
    hashes = [r['hash'] for r in results]
    unique_hashes = set(hashes)

    # Within a single run, all hashes should be unique (no duplicates)
    assert len(hashes) == len(unique_hashes), (
        f"Found {len(hashes) - len(unique_hashes)} duplicate hashes within single run"
    )
    print(f"Within-run dedup: {len(results)} results, all unique hashes. Passed!")

    # Cross-run dedup is handled by cursor byte_offset in run_extraction(),
    # verified in test_file_extraction() below.


def test_cursor_roundtrip():
    """Test cursor file save/load."""
    from extract_memories import load_cursor, save_cursor, cleanup_cursor, CURSOR_FILE

    test_session = "test-session-12345"
    cursor = {
        "byte_offset": 1000,
        "extracted_hashes": ["abc123", "def456"],
        "last_run": "2026-01-30T12:00:00",
    }

    save_cursor(test_session, cursor)
    loaded = load_cursor(test_session)

    assert loaded["byte_offset"] == 1000, f"Expected 1000, got {loaded['byte_offset']}"
    assert len(loaded["extracted_hashes"]) == 2
    print("Cursor roundtrip test passed!")

    # Cleanup
    cleanup_cursor(test_session)
    loaded_after = load_cursor(test_session)
    assert loaded_after["byte_offset"] == 0, "Should be reset after cleanup"
    print("Cursor cleanup test passed!")


def test_file_extraction():
    """Test extraction from an actual file with cursor tracking."""
    # Write sample transcript to a temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(SAMPLE_TRANSCRIPT)
        temp_path = f.name

    try:
        results = run_extraction(
            session_id="test-file-session",
            transcript_path=temp_path,
            project_path="C:\\test\\project",
            is_session_end=False,
        )
        print(f"\nFile extraction results: {json.dumps(results, indent=2)}")
        assert results["extracted"] > 0, "Should have extracted something"
        print("File extraction test passed!")

        # Run again - should extract 0 due to cursor
        results2 = run_extraction(
            session_id="test-file-session",
            transcript_path=temp_path,
            project_path="C:\\test\\project",
            is_session_end=True,  # Clean up cursor
        )
        print(f"Second run results: {json.dumps(results2, indent=2)}")
        # The stored count might be 0 due to API being unavailable, but extracted should be 0
        # because cursor says we already processed this content
        assert results2["extracted"] == 0, f"Expected 0 new extractions, got {results2['extracted']}"
        print("Cursor tracking test passed!")

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing extract_memories.py")
    print("=" * 60)

    test_extraction()
    print()
    test_dedup()
    print()
    test_cursor_roundtrip()
    print()
    test_file_extraction()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
