"""Start the memory agent as a proper background daemon on Windows."""
import subprocess
import sys
import os
import time

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(AGENT_DIR, "memory-agent.log")

def is_running():
    """Check if agent is already running."""
    try:
        import requests
        r = requests.get("http://localhost:8100/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def start_daemon():
    """Start the memory agent as a detached background process."""
    if is_running():
        print("Memory agent is already running!")
        return True

    # Windows-specific flags
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
        print(f"Started memory agent (PID: {proc.pid})")

    # Wait for startup
    for i in range(10):
        time.sleep(0.5)
        if is_running():
            print("Memory agent is now running!")
            return True

    print("Warning: Agent started but health check failed. Check log file.")
    return False

if __name__ == "__main__":
    start_daemon()
