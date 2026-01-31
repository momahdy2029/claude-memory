"""
Claude Memory Manager - Windows GUI Application
A system tray application to manage the Claude Memory Agent server.

Features:
- Start/Stop/Restart the Memory Agent
- Live log viewer with auto-scroll
- System tray integration (minimize to tray)
- Windows startup registration
- Ollama status monitoring
- Dark themed UI
"""

import os
import sys
import json
import socket
import subprocess
import threading
import webbrowser
import winreg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path
from typing import Optional
import queue
import time

# Configuration
APP_NAME = "Claude Memory Manager"
APP_VERSION = "1.0.0"
AGENT_DIR = Path(__file__).parent.absolute()
VENV_DIR = AGENT_DIR / "venv"
PYTHON_EXE = VENV_DIR / "Scripts" / "python.exe"
MAIN_SCRIPT = AGENT_DIR / "main.py"
SETTINGS_FILE = AGENT_DIR / "manager_settings.json"

# Load port from environment or use default
from dotenv import load_dotenv
load_dotenv(AGENT_DIR / ".env")
PORT = int(os.getenv("PORT", "8102"))
DASHBOARD_URL = os.getenv("MEMORY_AGENT_URL", f"http://localhost:{PORT}") + "/dashboard"

# Color scheme (Dark theme matching dashboard)
COLORS = {
    "bg": "#1a1a2e",
    "panel": "#16213e",
    "accent": "#0f3460",
    "text": "#e6e6e6",
    "text_dim": "#8888aa",
    "success": "#00ff88",
    "error": "#ff4444",
    "warning": "#ffaa00",
    "button_hover": "#1f4068",
    "button_bg": "#0f3460",
    "border": "#2a2a4e",
    "disabled_fg": "#666688",
    "disabled_bg": "#0a0a1e",
}


class Settings:
    """Manages persistent settings."""

    def __init__(self):
        self.run_on_startup = False
        self.auto_start_agent = True
        self.minimize_to_tray = True
        self.start_minimized = False
        self.load()

    def load(self):
        """Load settings from file."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    self.run_on_startup = data.get("run_on_startup", False)
                    self.auto_start_agent = data.get("auto_start_agent", True)
                    self.minimize_to_tray = data.get("minimize_to_tray", True)
                    self.start_minimized = data.get("start_minimized", False)
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save(self):
        """Save settings to file."""
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump({
                    "run_on_startup": self.run_on_startup,
                    "auto_start_agent": self.auto_start_agent,
                    "minimize_to_tray": self.minimize_to_tray,
                    "start_minimized": self.start_minimized,
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")


class StartupManager:
    """Manages Windows startup registry entries."""

    REGISTRY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
    APP_KEY_NAME = "ClaudeMemoryManager"

    @classmethod
    def is_registered(cls) -> bool:
        """Check if app is registered for startup."""
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, cls.REGISTRY_KEY, 0, winreg.KEY_READ)
            try:
                winreg.QueryValueEx(key, cls.APP_KEY_NAME)
                return True
            except FileNotFoundError:
                return False
            finally:
                winreg.CloseKey(key)
        except Exception:
            return False

    @classmethod
    def register(cls):
        """Add app to Windows startup."""
        try:
            # Get the path to pythonw.exe (no console) and manager.py
            pythonw = VENV_DIR / "Scripts" / "pythonw.exe"
            if not pythonw.exists():
                pythonw = PYTHON_EXE  # Fallback to python.exe

            startup_cmd = f'"{pythonw}" "{AGENT_DIR / "manager.py"}"'

            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, cls.REGISTRY_KEY, 0, winreg.KEY_WRITE)
            winreg.SetValueEx(key, cls.APP_KEY_NAME, 0, winreg.REG_SZ, startup_cmd)
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"Error registering startup: {e}")
            return False

    @classmethod
    def unregister(cls):
        """Remove app from Windows startup."""
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, cls.REGISTRY_KEY, 0, winreg.KEY_WRITE)
            try:
                winreg.DeleteValue(key, cls.APP_KEY_NAME)
            except FileNotFoundError:
                pass  # Already not registered
            winreg.CloseKey(key)
            return True
        except Exception as e:
            print(f"Error unregistering startup: {e}")
            return False


class ProcessManager:
    """Manages the Memory Agent process."""

    def __init__(self, log_callback=None):
        self.process: Optional[subprocess.Popen] = None
        self.log_callback = log_callback
        self.log_queue = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.running = False

    def is_port_in_use(self) -> bool:
        """Check if the port is already in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", PORT)) == 0
        except Exception:
            return False

    def is_running(self) -> bool:
        """Check if the Memory Agent is actually running (not just port in use)."""
        if self.process and self.process.poll() is None:
            return True

        # Actually check if Memory Agent API responds, not just port
        try:
            import urllib.request
            req = urllib.request.Request(
                f"http://localhost:{PORT}/api/stats",
                headers={"User-Agent": "MemoryManager/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    return True
        except:
            pass

        return False

    def _kill_port_processes(self):
        """Kill all processes using our port."""
        try:
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            for line in result.stdout.split("\n"):
                if f":{PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit():
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW
                            )
        except:
            pass

    def start(self) -> tuple[bool, str]:
        """Start the Memory Agent."""
        if self.is_running():
            return False, "Agent is already running"

        # If port is in use but agent isn't responding, kill zombie processes first
        if self.is_port_in_use():
            self._kill_port_processes()
            time.sleep(1)

        if not PYTHON_EXE.exists():
            return False, f"Python not found at {PYTHON_EXE}"

        if not MAIN_SCRIPT.exists():
            return False, f"Main script not found at {MAIN_SCRIPT}"

        try:
            # Create the process with pipes for output
            self.process = subprocess.Popen(
                [str(PYTHON_EXE), str(MAIN_SCRIPT)],
                cwd=str(AGENT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW
            )

            self.running = True

            # Start log reader thread
            self.reader_thread = threading.Thread(target=self._read_logs, daemon=True)
            self.reader_thread.start()

            # Wait a bit and check if it started
            time.sleep(1.5)
            if self.process.poll() is not None:
                # Process exited, read any error output
                return False, "Process exited immediately - check logs for details"

            return True, f"Agent started successfully on port {PORT}"

        except Exception as e:
            return False, f"Failed to start: {str(e)}"

    def stop(self) -> tuple[bool, str]:
        """Stop the Memory Agent."""
        self.running = False

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                self.process = None
                return True, "Agent stopped gracefully"
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process = None
                return True, "Agent killed (forced termination)"
            except Exception as e:
                return False, f"Error stopping: {str(e)}"

        # Kill ALL processes on the port (not just one)
        killed_pids = []
        try:
            # Find ALL processes using the port
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            pids_to_kill = set()
            for line in result.stdout.split("\n"):
                if f":{PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit():
                            pids_to_kill.add(pid)

            # Kill ALL of them at once
            for pid in pids_to_kill:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    killed_pids.append(pid)
                except:
                    pass
        except Exception as e:
            pass

        # Also kill any stray python processes running memory-agent as fallback
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force"],
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5
            )
        except:
            pass

        if killed_pids:
            return True, f"Killed {len(killed_pids)} process(es) (PIDs: {', '.join(killed_pids)})"
        return True, "Agent stopped"

    def restart(self) -> tuple[bool, str]:
        """Restart the Memory Agent."""
        stop_success, stop_msg = self.stop()

        # Wait until port is actually free (max 10 seconds)
        for _ in range(20):
            if not self.is_port_in_use():
                break
            time.sleep(0.5)
        else:
            # Force kill everything one more time
            self.stop()
            time.sleep(1)

        start_success, start_msg = self.start()

        if start_success:
            return True, f"Agent restarted: {start_msg}"
        else:
            return False, f"Restart failed: {start_msg}"

    def _read_logs(self):
        """Read logs from the process output."""
        try:
            while self.running and self.process and self.process.stdout:
                try:
                    line = self.process.stdout.readline()
                    if line:
                        self.log_queue.put(line.rstrip())
                    elif self.process.poll() is not None:
                        break
                except Exception:
                    break
        except Exception as e:
            self.log_queue.put(f"[Log reader error: {e}]")

    def get_logs(self) -> list[str]:
        """Get all pending log lines."""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs


def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(("localhost", 11434)) == 0
    except Exception:
        return False


class ClaudeMemoryManagerApp:
    """Main application window."""

    def __init__(self):
        self.settings = Settings()
        self.process_manager = ProcessManager()
        self.root = tk.Tk()
        self.is_minimized = False

        self.setup_window()
        self.create_widgets()
        self.start_log_updater()
        self.update_status()
        self.sync_startup_checkbox()

        # Initial log message
        self.log(f"Claude Memory Manager v{APP_VERSION}", "info")
        self.log(f"Agent directory: {AGENT_DIR}", "info")

        # Check Ollama on startup
        if not check_ollama_running():
            self.log("WARNING: Ollama is not running.", "warning")
            self.log("The Memory Agent requires Ollama for embeddings.", "warning")
            self.log("Please start Ollama before starting the agent.", "warning")
        else:
            self.log("Ollama is running and ready.", "success")

        # Auto-start if configured
        if self.settings.auto_start_agent and not self.process_manager.is_running():
            self.root.after(1000, self.start_agent)

        # Start minimized if configured
        if self.settings.start_minimized:
            self.root.after(100, self.minimize_window)

    def setup_window(self):
        """Configure the main window."""
        self.root.title(APP_NAME)
        self.root.geometry("580x650")
        self.root.minsize(480, 550)
        self.root.configure(bg=COLORS["bg"])

        # Center the window
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")

        # Configure styles
        style = ttk.Style()
        style.theme_use("clam")

        # Configure ttk styles
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("TCheckbutton", background=COLORS["bg"], foreground=COLORS["text"])

        # Handle window close and minimize
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Unmap>", self.on_minimize)

    def create_widgets(self):
        """Create all UI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header with title and version
        header_frame = tk.Frame(main_frame, bg=COLORS["bg"])
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = tk.Label(
            header_frame,
            text="Claude Memory Manager",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["text"]
        )
        title_label.pack(side=tk.LEFT)

        version_label = tk.Label(
            header_frame,
            text=f"v{APP_VERSION}",
            font=("Segoe UI", 10),
            bg=COLORS["bg"],
            fg=COLORS["text_dim"]
        )
        version_label.pack(side=tk.LEFT, padx=(10, 0), pady=(8, 0))

        # Status panel
        status_frame = tk.Frame(main_frame, bg=COLORS["panel"], padx=20, pady=15)
        status_frame.pack(fill=tk.X, pady=(0, 15))

        # Left side - Agent status
        agent_status_frame = tk.Frame(status_frame, bg=COLORS["panel"])
        agent_status_frame.pack(side=tk.LEFT)

        status_label = tk.Label(
            agent_status_frame,
            text="Agent Status:",
            font=("Segoe UI", 11),
            bg=COLORS["panel"],
            fg=COLORS["text_dim"]
        )
        status_label.pack(side=tk.LEFT)

        self.status_indicator = tk.Label(
            agent_status_frame,
            text="\u25cf",  # Filled circle
            font=("Segoe UI", 16),
            bg=COLORS["panel"],
            fg=COLORS["error"]
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(8, 5))

        self.status_text = tk.Label(
            agent_status_frame,
            text="Stopped",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["text"]
        )
        self.status_text.pack(side=tk.LEFT)

        # Right side - Ollama status
        self.ollama_frame = tk.Frame(status_frame, bg=COLORS["panel"])
        self.ollama_frame.pack(side=tk.RIGHT)

        ollama_label = tk.Label(
            self.ollama_frame,
            text="Ollama:",
            font=("Segoe UI", 10),
            bg=COLORS["panel"],
            fg=COLORS["text_dim"]
        )
        ollama_label.pack(side=tk.LEFT)

        self.ollama_indicator = tk.Label(
            self.ollama_frame,
            text="\u25cf",
            font=("Segoe UI", 12),
            bg=COLORS["panel"],
            fg=COLORS["error"]
        )
        self.ollama_indicator.pack(side=tk.LEFT, padx=(5, 3))

        self.ollama_status = tk.Label(
            self.ollama_frame,
            text="Stopped",
            font=("Segoe UI", 10),
            bg=COLORS["panel"],
            fg=COLORS["text_dim"]
        )
        self.ollama_status.pack(side=tk.LEFT)

        # Control buttons panel
        buttons_frame = tk.Frame(main_frame, bg=COLORS["bg"])
        buttons_frame.pack(fill=tk.X, pady=(0, 15))

        # Create styled buttons
        button_config = {
            "font": ("Segoe UI", 11),
            "width": 11,
            "cursor": "hand2",
            "relief": tk.FLAT,
            "bd": 0,
        }

        self.start_btn = tk.Button(
            buttons_frame,
            text="\u25b6  Start",
            bg=COLORS["button_bg"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            disabledforeground=COLORS["disabled_fg"],
            command=self.start_agent,
            **button_config
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10), ipady=8)
        self._add_hover_effect(self.start_btn)

        self.stop_btn = tk.Button(
            buttons_frame,
            text="\u25a0  Stop",
            bg=COLORS["button_bg"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            disabledforeground=COLORS["disabled_fg"],
            command=self.stop_agent,
            **button_config
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10), ipady=8)
        self._add_hover_effect(self.stop_btn)

        self.restart_btn = tk.Button(
            buttons_frame,
            text="\u21bb  Restart",
            bg=COLORS["button_bg"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            disabledforeground=COLORS["disabled_fg"],
            command=self.restart_agent,
            **button_config
        )
        self.restart_btn.pack(side=tk.LEFT, ipady=8)
        self._add_hover_effect(self.restart_btn)

        # Dashboard button
        dashboard_frame = tk.Frame(main_frame, bg=COLORS["bg"])
        dashboard_frame.pack(fill=tk.X, pady=(0, 15))

        self.dashboard_btn = tk.Button(
            dashboard_frame,
            text="\U0001F4CA  Open Dashboard",
            font=("Segoe UI", 12),
            bg=COLORS["accent"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            command=self.open_dashboard
        )
        self.dashboard_btn.pack(fill=tk.X, ipady=12)
        self._add_hover_effect(self.dashboard_btn)

        # Settings panel
        settings_frame = tk.Frame(main_frame, bg=COLORS["panel"], padx=15, pady=15)
        settings_frame.pack(fill=tk.X, pady=(0, 15))

        settings_title = tk.Label(
            settings_frame,
            text="Settings",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["panel"],
            fg=COLORS["text"]
        )
        settings_title.pack(anchor=tk.W, pady=(0, 10))

        # Checkbox styling
        checkbox_config = {
            "font": ("Segoe UI", 10),
            "bg": COLORS["panel"],
            "fg": COLORS["text"],
            "selectcolor": COLORS["accent"],
            "activebackground": COLORS["panel"],
            "activeforeground": COLORS["text"],
            "cursor": "hand2",
        }

        # Run on startup checkbox
        self.startup_var = tk.BooleanVar(value=self.settings.run_on_startup)
        self.startup_check = tk.Checkbutton(
            settings_frame,
            text="Run on Windows Startup",
            variable=self.startup_var,
            command=self.toggle_startup,
            **checkbox_config
        )
        self.startup_check.pack(anchor=tk.W)

        # Auto-start agent checkbox
        self.autostart_var = tk.BooleanVar(value=self.settings.auto_start_agent)
        self.autostart_check = tk.Checkbutton(
            settings_frame,
            text="Auto-start agent when manager launches",
            variable=self.autostart_var,
            command=self.toggle_autostart,
            **checkbox_config
        )
        self.autostart_check.pack(anchor=tk.W, pady=(5, 0))

        # Start minimized checkbox
        self.start_minimized_var = tk.BooleanVar(value=self.settings.start_minimized)
        self.start_minimized_check = tk.Checkbutton(
            settings_frame,
            text="Start minimized to taskbar",
            variable=self.start_minimized_var,
            command=self.toggle_start_minimized,
            **checkbox_config
        )
        self.start_minimized_check.pack(anchor=tk.W, pady=(5, 0))

        # Minimize to tray checkbox
        self.minimize_tray_var = tk.BooleanVar(value=self.settings.minimize_to_tray)
        self.minimize_tray_check = tk.Checkbutton(
            settings_frame,
            text="Keep agent running when window is closed",
            variable=self.minimize_tray_var,
            command=self.toggle_minimize_tray,
            **checkbox_config
        )
        self.minimize_tray_check.pack(anchor=tk.W, pady=(5, 0))

        # Logs section
        logs_header = tk.Frame(main_frame, bg=COLORS["bg"])
        logs_header.pack(fill=tk.X, pady=(0, 5))

        logs_label = tk.Label(
            logs_header,
            text="Logs",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["text"]
        )
        logs_label.pack(side=tk.LEFT)

        # Clear logs button (smaller, next to title)
        clear_btn = tk.Button(
            logs_header,
            text="Clear",
            font=("Segoe UI", 9),
            bg=COLORS["accent"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            cursor="hand2",
            relief=tk.FLAT,
            bd=0,
            command=self.clear_logs
        )
        clear_btn.pack(side=tk.RIGHT, ipadx=8, ipady=2)
        self._add_hover_effect(clear_btn)

        # Log viewer with border
        log_frame = tk.Frame(main_frame, bg=COLORS["border"], padx=1, pady=1)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg=COLORS["panel"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["accent"],
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure log text tags for coloring
        self.log_text.tag_configure("info", foreground=COLORS["text"])
        self.log_text.tag_configure("warning", foreground=COLORS["warning"])
        self.log_text.tag_configure("error", foreground=COLORS["error"])
        self.log_text.tag_configure("success", foreground=COLORS["success"])
        self.log_text.tag_configure("dim", foreground=COLORS["text_dim"])

        # Footer with info
        footer = tk.Label(
            main_frame,
            text=f"Port: {PORT}  |  Dashboard: {DASHBOARD_URL}",
            font=("Segoe UI", 9),
            bg=COLORS["bg"],
            fg=COLORS["text_dim"]
        )
        footer.pack(pady=(10, 0))

    def _add_hover_effect(self, button):
        """Add hover effect to a button."""
        original_bg = button.cget("bg")

        def on_enter(e):
            if str(button.cget("state")) != "disabled":
                button.configure(bg=COLORS["button_hover"])

        def on_leave(e):
            if str(button.cget("state")) != "disabled":
                button.configure(bg=original_bg)

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

    def log(self, message: str, tag: str = "info"):
        """Add a message to the log viewer."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", tag)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def clear_logs(self):
        """Clear the log viewer."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.log("Logs cleared", "dim")

    def update_status(self):
        """Update the status indicators."""
        is_running = self.process_manager.is_running()
        ollama_running = check_ollama_running()

        # Update agent status
        if is_running:
            self.status_indicator.configure(fg=COLORS["success"])
            self.status_text.configure(text="Running", fg=COLORS["success"])
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self.restart_btn.configure(state=tk.NORMAL)
        else:
            self.status_indicator.configure(fg=COLORS["error"])
            self.status_text.configure(text="Stopped", fg=COLORS["error"])
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self.restart_btn.configure(state=tk.DISABLED)

        # Update Ollama status
        if ollama_running:
            self.ollama_indicator.configure(fg=COLORS["success"])
            self.ollama_status.configure(text="Running", fg=COLORS["success"])
        else:
            self.ollama_indicator.configure(fg=COLORS["warning"])
            self.ollama_status.configure(text="Stopped", fg=COLORS["warning"])

        # Schedule next update
        self.root.after(2000, self.update_status)

    def start_log_updater(self):
        """Start the log updater loop."""
        def update_logs():
            logs = self.process_manager.get_logs()
            for line in logs:
                # Determine log level for coloring
                tag = "info"
                line_lower = line.lower()
                if "error" in line_lower or "exception" in line_lower or "traceback" in line_lower:
                    tag = "error"
                elif "warning" in line_lower or "warn" in line_lower:
                    tag = "warning"
                elif "started" in line_lower or "success" in line_lower or "ready" in line_lower:
                    tag = "success"
                elif "info:" in line_lower or "debug:" in line_lower:
                    tag = "dim"

                self.log(line, tag)

            self.root.after(100, update_logs)

        update_logs()

    def sync_startup_checkbox(self):
        """Sync the startup checkbox with actual registry state."""
        is_registered = StartupManager.is_registered()
        self.startup_var.set(is_registered)
        self.settings.run_on_startup = is_registered

    def start_agent(self):
        """Start the Memory Agent."""
        if not check_ollama_running():
            result = messagebox.askyesno(
                "Ollama Not Running",
                "Ollama is not running. The Memory Agent requires Ollama for embeddings.\n\n"
                "Do you want to start the agent anyway?",
                parent=self.root
            )
            if not result:
                return

        self.log("Starting Memory Agent...", "info")

        # Run in thread to avoid blocking UI
        def start_thread():
            success, message = self.process_manager.start()
            self.root.after(0, lambda: self.log(message, "success" if success else "error"))

        threading.Thread(target=start_thread, daemon=True).start()

    def stop_agent(self):
        """Stop the Memory Agent."""
        self.log("Stopping Memory Agent...", "info")

        def stop_thread():
            success, message = self.process_manager.stop()
            self.root.after(0, lambda: self.log(message, "success" if success else "error"))

        threading.Thread(target=stop_thread, daemon=True).start()

    def restart_agent(self):
        """Restart the Memory Agent."""
        self.log("Restarting Memory Agent...", "info")

        def restart_thread():
            success, message = self.process_manager.restart()
            self.root.after(0, lambda: self.log(message, "success" if success else "error"))

        threading.Thread(target=restart_thread, daemon=True).start()

    def open_dashboard(self):
        """Open the dashboard in the default browser."""
        if not self.process_manager.is_running():
            result = messagebox.askyesno(
                "Agent Not Running",
                "The Memory Agent is not running.\n\nWould you like to start it first?",
                parent=self.root
            )
            if result:
                self.start_agent()
                # Wait for agent to start, then open dashboard
                self.root.after(3000, lambda: webbrowser.open(DASHBOARD_URL))
                self.log("Will open dashboard after agent starts...", "info")
                return
            else:
                # Open anyway (might show error page)
                pass

        webbrowser.open(DASHBOARD_URL)
        self.log(f"Opened dashboard in browser", "info")

    def toggle_startup(self):
        """Toggle run on Windows startup."""
        enabled = self.startup_var.get()

        if enabled:
            if StartupManager.register():
                self.settings.run_on_startup = True
                self.log("Added to Windows startup", "success")
            else:
                self.startup_var.set(False)
                self.log("Failed to add to Windows startup", "error")
        else:
            if StartupManager.unregister():
                self.settings.run_on_startup = False
                self.log("Removed from Windows startup", "info")
            else:
                self.startup_var.set(True)
                self.log("Failed to remove from Windows startup", "error")

        self.settings.save()

    def toggle_autostart(self):
        """Toggle auto-start agent on launch."""
        self.settings.auto_start_agent = self.autostart_var.get()
        self.settings.save()
        status = "enabled" if self.settings.auto_start_agent else "disabled"
        self.log(f"Auto-start agent {status}", "info")

    def toggle_start_minimized(self):
        """Toggle start minimized setting."""
        self.settings.start_minimized = self.start_minimized_var.get()
        self.settings.save()
        status = "enabled" if self.settings.start_minimized else "disabled"
        self.log(f"Start minimized {status}", "info")

    def toggle_minimize_tray(self):
        """Toggle minimize to tray setting."""
        self.settings.minimize_to_tray = self.minimize_tray_var.get()
        self.settings.save()
        status = "enabled" if self.settings.minimize_to_tray else "disabled"
        self.log(f"Keep running on close {status}", "info")

    def minimize_window(self):
        """Minimize the window."""
        self.root.iconify()
        self.is_minimized = True

    def on_minimize(self, event):
        """Handle window minimize event."""
        if event.widget == self.root:
            self.is_minimized = True

    def on_close(self):
        """Handle window close."""
        if self.settings.minimize_to_tray and self.process_manager.is_running():
            # Just hide the window, keep agent running
            self.root.iconify()
            self.is_minimized = True
            self.log("Window minimized. Agent continues running.", "dim")
        else:
            # Actually close
            if self.process_manager.is_running():
                result = messagebox.askyesnocancel(
                    "Exit Confirmation",
                    "The Memory Agent is still running.\n\n"
                    "Yes = Stop agent and exit\n"
                    "No = Exit but keep agent running\n"
                    "Cancel = Don't exit",
                    parent=self.root
                )
                if result is True:
                    self.log("Stopping agent and exiting...", "info")
                    self.process_manager.stop()
                    self.root.destroy()
                elif result is False:
                    self.log("Exiting manager (agent will continue running)", "info")
                    self.root.destroy()
                # If result is None (Cancel), do nothing
            else:
                self.root.destroy()

    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    # Ensure we're in the right directory
    os.chdir(AGENT_DIR)

    # Create and run the application
    app = ClaudeMemoryManagerApp()
    app.run()


if __name__ == "__main__":
    main()
