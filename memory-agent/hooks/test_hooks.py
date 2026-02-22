#!/usr/bin/env python3
"""Test the hook wrapper scripts by simulating Claude Code's stdin JSON."""
import subprocess
import json
import sys
import os
import tempfile

HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_hook(script_name, hook_data, timeout=10):
    """Run a hook script with JSON data on stdin."""
    script_path = os.path.join(HOOKS_DIR, script_name)
    stdin_data = json.dumps(hook_data)

    result = subprocess.run(
        [PYTHON, script_path],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def test_precompact_no_transcript():
    """PreCompact with no transcript should exit 0 gracefully."""
    result = run_hook("pre_compact_hook.py", {
        "session_id": "test-precompact-1",
        "hook_event_name": "PreCompact",
    })
    print(f"  exit code: {result.returncode}")
    print(f"  stderr: {result.stderr.strip()}")
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
    print("  PASSED")


def test_precompact_with_transcript():
    """PreCompact with a real transcript file."""
    # Create a temp transcript
    transcript_text = (
        "User: Fix the database timeout bug\n"
        "Assistant: I found the root cause. The connection pool was exhausted.\n"
        "Error: ConnectionPoolExhausted after 30 seconds\n"
        "I decided to increase the pool size from 5 to 20 connections.\n"
        "Fixed the issue by also adding connection recycling every 300 seconds.\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(transcript_text)
        temp_path = f.name

    try:
        result = run_hook("pre_compact_hook.py", {
            "session_id": "test-precompact-2",
            "transcript_path": temp_path,
            "hook_event_name": "PreCompact",
            "cwd": os.path.dirname(HOOKS_DIR),
        })
        print(f"  exit code: {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
        assert "Extraction complete" in result.stderr, "Should report extraction results"
        print("  PASSED")
    finally:
        os.unlink(temp_path)


def test_session_end_no_transcript():
    """SessionEnd with no transcript should exit 0 gracefully."""
    result = run_hook("session_end_hook.py", {
        "session_id": "test-sessionend-1",
        "hook_event_name": "SessionEnd",
    })
    print(f"  exit code: {result.returncode}")
    print(f"  stderr: {result.stderr.strip()}")
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
    print("  PASSED")


def test_session_end_with_transcript():
    """SessionEnd with a real transcript file."""
    transcript_text = (
        "User: Set up the caching layer\n"
        "Assistant: Going with Redis for the caching layer.\n"
        "The pattern is to cache at the service layer, not the controller.\n"
        "Convention: All cache keys must be prefixed with the service name.\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(transcript_text)
        temp_path = f.name

    try:
        result = run_hook("session_end_hook.py", {
            "session_id": "test-sessionend-2",
            "transcript_path": temp_path,
            "hook_event_name": "SessionEnd",
            "cwd": os.path.dirname(HOOKS_DIR),
        })
        print(f"  exit code: {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}"
        assert "Extraction complete" in result.stderr or "SessionEnd" in result.stderr
        print("  PASSED")
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing hook wrapper scripts")
    print("=" * 60)

    print("\n1. PreCompact - no transcript:")
    test_precompact_no_transcript()

    print("\n2. PreCompact - with transcript:")
    test_precompact_with_transcript()

    print("\n3. SessionEnd - no transcript:")
    test_session_end_no_transcript()

    print("\n4. SessionEnd - with transcript:")
    test_session_end_with_transcript()

    print()
    print("=" * 60)
    print("ALL HOOK TESTS PASSED")
    print("=" * 60)
