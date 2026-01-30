"""Start the memory agent as a proper background daemon on Windows.

Uses msvcrt.locking() for a true Windows mutex to prevent multiple
simultaneous startup attempts. The server itself has its own mutex.
"""
import subprocess
import sys
import os
import time
import msvcrt

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(AGENT_DIR, "memory-agent.log")
STARTUP_LOCK_FILE = os.path.join(AGENT_DIR, "memory-agent-startup.lock")
PID_FILE = os.path.join(AGENT_DIR, "memory-agent.pid")

# Global handle - must stay open for lock to persist
_startup_lock_handle = None


def acquire_startup_lock() -> bool:
    """Acquire startup mutex using Windows file locking (msvcrt.locking).

    This prevents multiple hooks from trying to start the agent simultaneously.
    The lock is held until release_startup_lock() is called.
    """
    global _startup_lock_handle

    try:
        # Open/create the lock file
        _startup_lock_handle = open(STARTUP_LOCK_FILE, 'w+')

        # Try non-blocking exclusive lock
        try:
            msvcrt.locking(_startup_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        except (IOError, OSError):
            # Lock held by another process - they're already starting the agent
            _startup_lock_handle.close()
            _startup_lock_handle = None
            return False

        # We have the lock - write our PID for debugging
        _startup_lock_handle.seek(0)
        _startup_lock_handle.truncate()
        _startup_lock_handle.write(str(os.getpid()))
        _startup_lock_handle.flush()
        return True

    except Exception as e:
        print(f"[STARTUP] Failed to acquire lock: {e}")
        if _startup_lock_handle:
            try:
                _startup_lock_handle.close()
            except:
                pass
            _startup_lock_handle = None
        return False


def release_startup_lock():
    """Release the startup mutex."""
    global _startup_lock_handle

    try:
        if _startup_lock_handle:
            try:
                msvcrt.locking(_startup_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
            except:
                pass
            _startup_lock_handle.close()
            _startup_lock_handle = None
    except Exception:
        pass


def is_running():
    """Check if agent is already running via health endpoint."""
    try:
        import requests
        r = requests.get("http://localhost:8102/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def read_pid():
    """Read the PID from the PID file if it exists."""
    try:
        if os.path.exists(PID_FILE):
            with open(PID_FILE, 'r') as f:
                return int(f.read().strip())
    except Exception:
        pass
    return None


def start_daemon():
    """Start the memory agent as a detached background process."""
    # First check: is it already responding?
    if is_running():
        print("Memory agent is already running!")
        return True

    # Second check: try to acquire mutex lock
    if not acquire_startup_lock():
        # Another startup is in progress, wait for it
        print("Waiting for other startup to complete...")
        for i in range(10):
            time.sleep(0.5)
            if is_running():
                print("Memory agent started by another process!")
                return True
        print("Other startup failed or timed out")
        return False

    try:
        # Windows-specific flags for detached process
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        with open(LOG_FILE, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, "run_server.py"],
                cwd=AGENT_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
                close_fds=True
            )

            # Save PID for future reference
            with open(PID_FILE, 'w') as f:
                f.write(str(proc.pid))

            print(f"Started memory agent (PID: {proc.pid})")

        # Wait for startup with health check
        for i in range(10):
            time.sleep(0.5)
            if is_running():
                print("Memory agent is now running!")
                return True

        print("Warning: Agent started but health check failed. Check log file.")
        return False

    finally:
        # Always release lock when done (success or failure)
        release_startup_lock()


if __name__ == "__main__":
    start_daemon()
