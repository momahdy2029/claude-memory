"""Run the memory agent server (for background/production use).

Uses file locking for a true process mutex:
- Windows: msvcrt.locking()
- macOS/Linux: fcntl.flock()
The lock is held for the entire lifetime of the server, ensuring only
one instance can run at a time.
"""
import os
import sys
import time
import atexit
import signal
import platform
import uvicorn
from dotenv import load_dotenv

IS_WINDOWS = platform.system() == "Windows"

load_dotenv()

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCK_FILE = os.path.join(AGENT_DIR, "memory-agent.lock")
PID_FILE = os.path.join(AGENT_DIR, "memory-agent.pid")
PORT = int(os.getenv("PORT", 8102))

# Global lock file handle - must stay open for lock to persist
_lock_handle = None


def is_port_in_use(port: int) -> bool:
    """Check if the port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def acquire_server_lock() -> bool:
    """Acquire exclusive server lock using platform-appropriate file locking.

    - Windows: msvcrt.locking() with LK_NBLCK for non-blocking exclusive lock
    - macOS/Linux: fcntl.flock() with LOCK_EX | LOCK_NB for non-blocking exclusive lock

    The lock is held as long as the file handle remains open.
    """
    global _lock_handle

    my_pid = os.getpid()

    try:
        # Open file for read/write, create if doesn't exist
        _lock_handle = open(LOCK_FILE, 'w+')

        # Try to acquire exclusive lock (non-blocking), platform-specific
        try:
            if IS_WINDOWS:
                import msvcrt
                msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            # Lock is held by another process
            print(f"[MUTEX] Cannot acquire lock - another instance is running")
            _lock_handle.close()
            _lock_handle = None
            return False

        # We have the lock - write our PID
        _lock_handle.seek(0)
        _lock_handle.truncate()
        _lock_handle.write(str(my_pid))
        _lock_handle.flush()

        # Double-check the port isn't somehow in use
        if is_port_in_use(PORT):
            print(f"[MUTEX] Port {PORT} is already in use!")
            release_server_lock()
            return False

        print(f"[MUTEX] Acquired server lock (PID: {my_pid})")
        return True

    except Exception as e:
        print(f"[MUTEX] Failed to acquire lock: {e}")
        if _lock_handle:
            try:
                _lock_handle.close()
            except:
                pass
            _lock_handle = None
        return False


def release_server_lock():
    """Release the server lock on exit."""
    global _lock_handle

    try:
        if _lock_handle:
            try:
                # Unlock the file, platform-specific
                if IS_WINDOWS:
                    import msvcrt
                    msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_UN)
            except:
                pass
            _lock_handle.close()
            _lock_handle = None
            print("[MUTEX] Released server lock")
    except Exception as e:
        print(f"[MUTEX] Error releasing lock: {e}")


def cleanup_and_exit(signum=None, frame=None):
    """Clean up lock and exit."""
    release_server_lock()
    if signum:
        sys.exit(0)


if __name__ == "__main__":
    # Register cleanup handlers FIRST
    atexit.register(release_server_lock)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    signal.signal(signal.SIGINT, cleanup_and_exit)

    # Try to acquire the mutex - this blocks other instances
    if not acquire_server_lock():
        print("[MUTEX] Cannot start: another instance is running")
        sys.exit(1)

    # Write PID file (for convenience, the lock is the real mutex)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    print(f"[SERVER] Starting memory agent on port {PORT}...")

    # Note: The lock is held because _lock_handle stays open
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=PORT,
        reload=False,
        log_level="warning"
    )
