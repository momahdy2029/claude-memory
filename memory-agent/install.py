#!/usr/bin/env python3
"""
Claude Memory Agent - Installation Script

This script sets up the Claude Memory Agent for first-time use:
1. Creates .env file with auto-detected paths
2. Configures Claude Code MCP settings
3. Sets up hooks for auto-start and context injection
4. Creates platform-specific startup scripts
5. Installs Python dependencies

Usage:
    python install.py              # Interactive installation
    python install.py --auto       # Auto-install with defaults
    python install.py --uninstall  # Remove Claude Code integration
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# Agent directory (where this script lives)
AGENT_DIR = Path(__file__).parent.resolve()

# Default configuration
DEFAULT_CONFIG = {
    "PORT": "8102",
    "HOST": "0.0.0.0",
    "MEMORY_AGENT_URL": "http://localhost:8102",
    "OLLAMA_HOST": "http://localhost:11434",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "LOG_LEVEL": "INFO",
    "USE_VECTOR_INDEX": "true",
    "DB_POOL_SIZE": "5",
    "DB_TIMEOUT": "30.0",
    "AUTH_ENABLED": "false",
}

# Claude Code settings paths
def get_claude_settings_dir() -> Path:
    """Get the Claude Code settings directory."""
    if sys.platform == "win32":
        return Path.home() / ".claude"
    elif sys.platform == "darwin":
        return Path.home() / ".claude"
    else:  # Linux
        return Path.home() / ".claude"

def get_claude_settings_file() -> Path:
    """Get the Claude Code settings.json file path."""
    return get_claude_settings_dir() / "settings.json"

def get_hooks_dir() -> Path:
    """Get the Claude Code hooks directory."""
    return get_claude_settings_dir() / "hooks"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, total: int, text: str):
    """Print a step indicator."""
    print(f"[{step}/{total}] {text}")


def print_success(text: str):
    """Print a success message."""
    print(f"  [OK] {text}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"  [!!] {text}")


def print_error(text: str):
    """Print an error message."""
    print(f"  [ERROR] {text}")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no answer."""
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'")


def prompt_value(question: str, default: str) -> str:
    """Prompt for a value with a default."""
    answer = input(f"{question} [{default}]: ").strip()
    return answer if answer else default


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 9):
        print_error(f"Python 3.9+ required, found {major}.{minor}")
        return False
    print_success(f"Python {major}.{minor} detected")
    return True


def check_nodejs() -> tuple[bool, Optional[str]]:
    """Check if Node.js is installed and get version."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Node.js {version} detected")
            return True, version
    except FileNotFoundError:
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    print_warning("Node.js not found")
    return False, None


def check_npm() -> bool:
    """Check if npm is available."""
    # On Windows, npm is a .cmd file
    commands_to_try = ["npm", "npm.cmd"] if sys.platform == "win32" else ["npm"]

    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=(sys.platform == "win32")  # Use shell on Windows
            )
            if result.returncode == 0:
                print_success(f"npm {result.stdout.strip()} detected")
                return True
        except Exception:
            continue
    return False


def check_claude_code() -> tuple[bool, Optional[str]]:
    """Check if Claude Code CLI is installed."""
    # Try different possible command names
    # On Windows, these may be .cmd files
    if sys.platform == "win32":
        commands_to_try = ["claude", "claude.cmd", "claude-code", "claude-code.cmd"]
    else:
        commands_to_try = ["claude", "claude-code"]

    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=(sys.platform == "win32")  # Use shell on Windows
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print_success(f"Claude Code detected: {version}")
                return True, version
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            continue
        except Exception:
            continue

    # Check if .claude directory exists (indicates Claude Code was used)
    claude_dir = get_claude_settings_dir()
    if claude_dir.exists():
        print_success("Claude Code settings directory found (~/.claude)")
        return True, "directory exists"

    print_warning("Claude Code not detected")
    return False, None


def install_claude_code() -> bool:
    """Attempt to install Claude Code via npm."""
    print("\nClaude Code is not installed. Installing via npm...")

    # On Windows, use npm.cmd via shell
    npm_cmd = "npm" if sys.platform != "win32" else "npm.cmd"

    try:
        result = subprocess.run(
            [npm_cmd, "install", "-g", "@anthropic-ai/claude-code"],
            capture_output=True,
            text=True,
            timeout=120,
            shell=(sys.platform == "win32")
        )
        if result.returncode == 0:
            print_success("Claude Code installed successfully!")
            print("  Run 'claude' to start Claude Code")
            return True
        else:
            print_error(f"npm install failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Installation timed out")
        return False
    except Exception as e:
        print_error(f"Installation failed: {e}")
        return False


def print_installation_instructions():
    """Print manual installation instructions for missing dependencies."""
    print("\n" + "="*60)
    print("  INSTALLATION REQUIRED")
    print("="*60)
    print("""
To use Claude Memory Agent, you need:

1. NODE.JS (required for Claude Code)
   Download from: https://nodejs.org/
   - Windows: Download and run the installer
   - Mac: brew install node
   - Linux: sudo apt install nodejs npm

2. CLAUDE CODE (the AI coding assistant)
   After installing Node.js, run:
   npm install -g @anthropic-ai/claude-code

3. OLLAMA (for embeddings - optional but recommended)
   Download from: https://ollama.ai/
   Then run: ollama pull nomic-embed-text

After installing the prerequisites, run this installer again:
   python install.py
""")


def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            print_success("Ollama is running")
            return True
    except Exception:
        pass

    print_warning("Ollama not detected")
    print("")
    print("  " + "="*56)
    print("  OLLAMA REQUIRED FOR SEMANTIC SEARCH")
    print("  " + "="*56)
    print("")
    print("  The memory agent needs Ollama for embeddings.")
    print("  Without it, semantic search will not work.")
    print("")
    print("  To install Ollama:")
    print("    1. Download from: https://ollama.ai/download")
    print("    2. Install and run: ollama pull nomic-embed-text")
    print("    3. Start Ollama: ollama serve")
    print("    4. Re-run this installer")
    print("")
    return False


def check_ollama_model(model: str = "nomic-embed-text") -> bool:
    """Check if the embedding model is available in Ollama."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = r.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if model in model_names:
                print_success(f"Embedding model '{model}' is available")
                return True
            print_warning(f"Model '{model}' not found. Run: ollama pull {model}")
    except Exception:
        pass
    return False


# =============================================================================
# INSTALLATION STEPS
# =============================================================================

def install_dependencies() -> bool:
    """Install Python dependencies from requirements.txt."""
    requirements_file = AGENT_DIR / "requirements.txt"
    if not requirements_file.exists():
        print_warning("requirements.txt not found, skipping dependency installation")
        return True

    print("Installing Python dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "-q"],
            check=True,
            capture_output=True
        )
        print_success("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e.stderr.decode()}")
        return False


def create_env_file(config: Dict[str, str], force: bool = False) -> bool:
    """Create the .env configuration file."""
    env_file = AGENT_DIR / ".env"

    if env_file.exists() and not force:
        if not prompt_yes_no(".env file already exists. Overwrite?", default=False):
            print_success("Keeping existing .env file")
            return True

    # Build .env content
    lines = [
        "# Claude Memory Agent Configuration",
        "# Generated by install.py",
        f"# Installation date: {__import__('datetime').datetime.now().isoformat()}",
        "",
        "# Server Configuration",
        f"HOST={config['HOST']}",
        f"PORT={config['PORT']}",
        f"MEMORY_AGENT_URL={config['MEMORY_AGENT_URL']}",
        "",
        "# Ollama Configuration",
        f"OLLAMA_HOST={config['OLLAMA_HOST']}",
        f"EMBEDDING_MODEL={config['EMBEDDING_MODEL']}",
        "",
        "# Database Configuration",
        f"DATABASE_PATH={AGENT_DIR / 'memories.db'}",
        f"USE_VECTOR_INDEX={config['USE_VECTOR_INDEX']}",
        f"DB_POOL_SIZE={config['DB_POOL_SIZE']}",
        f"DB_TIMEOUT={config['DB_TIMEOUT']}",
        "",
        "# Logging",
        f"LOG_LEVEL={config['LOG_LEVEL']}",
        "",
        "# Authentication (disabled by default for local use)",
        f"AUTH_ENABLED={config['AUTH_ENABLED']}",
    ]

    try:
        env_file.write_text("\n".join(lines) + "\n")
        print_success(f"Created .env file at {env_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create .env file: {e}")
        return False


def create_startup_script() -> bool:
    """Create platform-specific startup script."""
    if sys.platform == "win32":
        return create_windows_startup_script()
    else:
        return create_unix_startup_script()


def create_windows_startup_script() -> bool:
    """Create Windows VBS startup script with auto-detected paths."""
    vbs_file = AGENT_DIR / "start-memory-agent.vbs"

    # Use forward slashes for VBS string, then replace
    agent_dir_str = str(AGENT_DIR).replace("\\", "\\\\")

    content = f'''' Start Memory Agent silently in background
' Auto-generated by install.py

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get the directory where this script is located
scriptPath = WScript.ScriptFullName
agentDir = fso.GetParentFolderName(scriptPath)

' Alternatively, use the configured path (uncomment if needed):
' agentDir = "{agent_dir_str}"

pythonCmd = "python """ & agentDir & "\\main.py"""

WshShell.CurrentDirectory = agentDir
WshShell.Run "cmd /c " & pythonCmd, 0, False
'''

    try:
        vbs_file.write_text(content)
        print_success(f"Created Windows startup script: {vbs_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create startup script: {e}")
        return False


def create_unix_startup_script() -> bool:
    """Create Unix/Mac startup script."""
    sh_file = AGENT_DIR / "start-memory-agent.sh"

    content = f'''#!/bin/bash
# Start Memory Agent in background
# Auto-generated by install.py

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

cd "$SCRIPT_DIR"
nohup python main.py > memory-agent.log 2>&1 &
echo "Memory Agent started (PID: $!)"
'''

    try:
        sh_file.write_text(content)
        sh_file.chmod(0o755)
        print_success(f"Created Unix startup script: {sh_file}")
        return True
    except Exception as e:
        print_error(f"Failed to create startup script: {e}")
        return False


def configure_claude_mcp(config: Dict[str, str]) -> bool:
    """Configure Claude Code MCP settings."""
    settings_file = get_claude_settings_file()
    settings_dir = get_claude_settings_dir()

    # Ensure settings directory exists
    settings_dir.mkdir(parents=True, exist_ok=True)

    # Load existing settings or create new
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
        except json.JSONDecodeError:
            print_warning("Existing settings.json is invalid, creating backup")
            shutil.copy(settings_file, settings_file.with_suffix(".json.bak"))
            settings = {}
    else:
        settings = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in settings:
        settings["mcpServers"] = {}

    # Add/update claude-memory server configuration
    settings["mcpServers"]["claude-memory"] = {
        "command": sys.executable,
        "args": [str(AGENT_DIR / "main.py")],
        "env": {
            "MEMORY_AGENT_URL": config["MEMORY_AGENT_URL"],
            "PORT": config["PORT"],
        }
    }

    try:
        settings_file.write_text(json.dumps(settings, indent=2))
        print_success(f"Configured Claude Code MCP settings: {settings_file}")
        return True
    except Exception as e:
        print_error(f"Failed to configure MCP settings: {e}")
        return False


def setup_hooks(config: Dict[str, str]) -> bool:
    """Set up Claude Code hooks for auto-start and context injection."""
    hooks_dir = get_hooks_dir()
    hooks_dir.mkdir(parents=True, exist_ok=True)

    source_hooks_dir = AGENT_DIR / "hooks"
    if not source_hooks_dir.exists():
        print_warning("Hooks directory not found in agent, skipping hook setup")
        return True

    # Hooks to install
    hooks_to_install = [
        "session_start.py",
        "session_end.py",
        "grounding-hook.py",
    ]

    installed = 0
    for hook_name in hooks_to_install:
        source = source_hooks_dir / hook_name
        if not source.exists():
            continue

        dest = hooks_dir / hook_name

        # Read source and update MEMORY_AGENT_URL default
        content = source.read_text()

        # Copy to hooks directory
        try:
            dest.write_text(content)
            installed += 1
        except Exception as e:
            print_warning(f"Failed to install hook {hook_name}: {e}")

    if installed > 0:
        print_success(f"Installed {installed} hooks to {hooks_dir}")

    return True


def configure_hooks_json() -> bool:
    """Configure hooks.json to enable the hooks."""
    hooks_file = get_claude_settings_dir() / "hooks.json"

    # Default hooks configuration
    hooks_config = {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "command": f"{sys.executable} {get_hooks_dir() / 'session_start.py'}",
                    "description": "Initialize memory session",
                    "timeout": 5000
                },
                {
                    "command": f"{sys.executable} {get_hooks_dir() / 'grounding-hook.py'}",
                    "description": "Inject grounding context",
                    "timeout": 3000
                }
            ],
            "SessionEnd": [
                {
                    "command": f"{sys.executable} {get_hooks_dir() / 'session_end.py'}",
                    "description": "Save session summary",
                    "timeout": 10000
                }
            ]
        }
    }

    # Merge with existing if present
    if hooks_file.exists():
        try:
            existing = json.loads(hooks_file.read_text())
            # Don't overwrite if user has customized
            if prompt_yes_no("hooks.json exists. Update with memory agent hooks?", default=True):
                if "hooks" not in existing:
                    existing["hooks"] = {}
                existing["hooks"].update(hooks_config["hooks"])
                hooks_config = existing
            else:
                print_success("Keeping existing hooks.json")
                return True
        except json.JSONDecodeError:
            pass

    try:
        hooks_file.write_text(json.dumps(hooks_config, indent=2))
        print_success(f"Configured hooks: {hooks_file}")
        return True
    except Exception as e:
        print_error(f"Failed to configure hooks: {e}")
        return False


def fix_agent_card_port() -> bool:
    """Fix the port in agent_card.py from 8100 to 8102."""
    agent_card_file = AGENT_DIR / "agent_card.py"

    if not agent_card_file.exists():
        return True

    content = agent_card_file.read_text()
    if '"url": "http://localhost:8100"' in content:
        content = content.replace(
            '"url": "http://localhost:8100"',
            '"url": "http://localhost:8102"'
        )
        agent_card_file.write_text(content)
        print_success("Fixed agent_card.py port (8100 -> 8102)")

    return True


def fix_dashboard_urls() -> bool:
    """Make dashboard.html use dynamic URLs."""
    dashboard_file = AGENT_DIR / "dashboard.html"

    if not dashboard_file.exists():
        return True

    content = dashboard_file.read_text()

    # Replace hardcoded URLs with dynamic detection
    old_js = "const API_URL = 'http://localhost:8102';\n        const WS_URL = 'ws://localhost:8102/ws';"
    new_js = """// Auto-detect server URL from current location
        const API_URL = window.location.origin || 'http://localhost:8102';
        const WS_URL = (window.location.protocol === 'https:' ? 'wss:' : 'ws:') + '//' + (window.location.host || 'localhost:8102') + '/ws';"""

    if old_js in content:
        content = content.replace(old_js, new_js)
        dashboard_file.write_text(content)
        print_success("Updated dashboard.html to use dynamic URLs")

    return True


def fix_start_daemon_url(config: Dict[str, str]) -> bool:
    """Fix hardcoded URL in start_daemon.py."""
    daemon_file = AGENT_DIR / "start_daemon.py"

    if not daemon_file.exists():
        return True

    content = daemon_file.read_text()

    # Replace hardcoded health check URL with environment variable
    old_line = 'r = requests.get("http://localhost:8102/health", timeout=2)'
    new_line = f'r = requests.get(os.getenv("MEMORY_AGENT_URL", "http://localhost:8102") + "/health", timeout=2)'

    if old_line in content:
        content = content.replace(old_line, new_line)
        daemon_file.write_text(content)
        print_success("Updated start_daemon.py to use environment variable")

    return True


def verify_installation() -> bool:
    """Verify the installation is working."""
    print("\nVerifying installation...")

    # Check .env exists
    if not (AGENT_DIR / ".env").exists():
        print_warning(".env file not created")
        return False

    # Try to import main module
    try:
        sys.path.insert(0, str(AGENT_DIR))
        from dotenv import load_dotenv
        load_dotenv(AGENT_DIR / ".env")
        print_success("Configuration loaded successfully")
    except Exception as e:
        print_warning(f"Could not load configuration: {e}")

    return True


def print_post_install_instructions(config: Dict[str, str]):
    """Print instructions for after installation."""
    print_header("Installation Complete!")

    print("Next steps:")
    print("")
    print("1. Make sure Ollama is running with the embedding model:")
    print(f"   ollama pull {config['EMBEDDING_MODEL']}")
    print(f"   ollama serve")
    print("")
    print("2. Start the Memory Agent:")
    print(f"   cd \"{AGENT_DIR}\"")
    print(f"   python main.py")
    print("")
    print("3. Or use the startup script:")
    if sys.platform == "win32":
        print(f"   Double-click: start-memory-agent.vbs")
    else:
        print(f"   ./start-memory-agent.sh")
    print("")
    print("4. Open the dashboard in your browser:")
    print(f"   {config['MEMORY_AGENT_URL']}/dashboard")
    print("")
    print("5. Restart Claude Code to load the MCP configuration")
    print("")
    print(f"Configuration file: {AGENT_DIR / '.env'}")
    print(f"Claude settings: {get_claude_settings_file()}")


# =============================================================================
# UNINSTALL
# =============================================================================

def uninstall() -> bool:
    """Remove Claude Code integration."""
    print_header("Uninstalling Claude Memory Agent Integration")

    # Remove MCP configuration
    settings_file = get_claude_settings_file()
    if settings_file.exists():
        try:
            settings = json.loads(settings_file.read_text())
            if "mcpServers" in settings and "claude-memory" in settings["mcpServers"]:
                del settings["mcpServers"]["claude-memory"]
                settings_file.write_text(json.dumps(settings, indent=2))
                print_success("Removed MCP configuration")
        except Exception as e:
            print_warning(f"Could not update settings: {e}")

    # Remove hooks
    hooks_dir = get_hooks_dir()
    hooks_to_remove = ["session_start.py", "session_end.py", "grounding-hook.py"]
    for hook in hooks_to_remove:
        hook_file = hooks_dir / hook
        if hook_file.exists():
            hook_file.unlink()
            print_success(f"Removed hook: {hook}")

    print("\nUninstall complete. The .env file and database are preserved.")
    print("To fully remove, delete the memory-agent directory.")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Install and configure Claude Memory Agent"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-install with default settings"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove Claude Code integration"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8102,
        help="Port for the memory agent (default: 8102)"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip installing Python dependencies"
    )
    parser.add_argument(
        "--skip-claude-check",
        action="store_true",
        help="Skip Claude Code installation check (for standalone use)"
    )

    args = parser.parse_args()

    if args.uninstall:
        return 0 if uninstall() else 1

    print_header("Claude Memory Agent Installation")
    print(f"Agent directory: {AGENT_DIR}")
    print(f"Platform: {sys.platform}")

    # Step 1: Check prerequisites
    total_steps = 9
    print_step(1, total_steps, "Checking prerequisites...")

    if not check_python_version():
        return 1

    # Check Node.js and Claude Code (unless skipped)
    claude_ok = False
    if not args.skip_claude_check:
        nodejs_ok, nodejs_version = check_nodejs()
        npm_ok = False
        if nodejs_ok:
            npm_ok = check_npm()

        # Check Claude Code
        claude_ok, claude_version = check_claude_code()

        # If Claude Code not found, check if we can install it
        if not claude_ok:
            if not nodejs_ok:
                # Neither Node.js nor Claude Code - show instructions and exit
                print_installation_instructions()
                return 1
            elif npm_ok:
                # Node.js available but Claude Code not installed
                if args.auto or prompt_yes_no("Claude Code not found. Install it now?"):
                    if not install_claude_code():
                        print_error("Could not install Claude Code automatically.")
                        print("Please install manually: npm install -g @anthropic-ai/claude-code")
                        if not prompt_yes_no("Continue anyway (memory agent only)?", default=False):
                            return 1
                    else:
                        claude_ok = True
                else:
                    print_warning("Skipping Claude Code installation")
                    print("  The memory agent will work, but Claude Code integration requires Claude Code")
            else:
                print_warning("npm not found - cannot auto-install Claude Code")
                print("  Install Claude Code manually: npm install -g @anthropic-ai/claude-code")
    else:
        print_success("Skipping Claude Code check (--skip-claude-check)")
        claude_ok = True  # Assume it's OK for standalone mode

    # Check Ollama
    ollama_ok = check_ollama()

    # Step 2: Configure settings
    print_step(2, total_steps, "Configuring settings...")

    config = DEFAULT_CONFIG.copy()
    config["PORT"] = str(args.port)
    config["MEMORY_AGENT_URL"] = f"http://localhost:{args.port}"

    if not args.auto:
        if not ollama_ok:
            config["OLLAMA_HOST"] = prompt_value(
                "Ollama host URL",
                config["OLLAMA_HOST"]
            )

        if prompt_yes_no("Use default embedding model (nomic-embed-text)?"):
            pass
        else:
            config["EMBEDDING_MODEL"] = prompt_value(
                "Embedding model name",
                config["EMBEDDING_MODEL"]
            )

    if ollama_ok:
        check_ollama_model(config["EMBEDDING_MODEL"])

    # Step 3: Install dependencies
    print_step(3, total_steps, "Installing dependencies...")
    if not args.skip_deps:
        install_dependencies()
    else:
        print_success("Skipped dependency installation")

    # Step 4: Create .env file
    print_step(4, total_steps, "Creating configuration file...")
    if not create_env_file(config, force=args.auto):
        return 1

    # Step 5: Fix hardcoded values
    print_step(5, total_steps, "Fixing hardcoded values...")
    fix_agent_card_port()
    fix_dashboard_urls()
    fix_start_daemon_url(config)

    # Step 6: Create startup script
    print_step(6, total_steps, "Creating startup script...")
    create_startup_script()

    # Step 7: Configure Claude Code (only if Claude Code is available)
    print_step(7, total_steps, "Configuring Claude Code integration...")

    if claude_ok:
        if args.auto or prompt_yes_no("Configure Claude Code MCP settings?"):
            configure_claude_mcp(config)

        if args.auto or prompt_yes_no("Install Claude Code hooks?"):
            setup_hooks(config)
            configure_hooks_json()
    else:
        print_warning("Skipping Claude Code configuration (Claude Code not installed)")
        print("  Run 'python install.py' again after installing Claude Code")

    # Step 8: Verify
    print_step(8, total_steps, "Verifying installation...")
    verify_installation()

    # Step 9: Auto-start agent if Ollama is ready
    print_step(9, total_steps, "Starting Memory Agent...")
    if ollama_ok:
        try:
            subprocess.run(
                [sys.executable, str(AGENT_DIR / "memory-agent"), "start"],
                cwd=str(AGENT_DIR),
                timeout=30
            )
            print_success("Memory Agent started!")
        except Exception as e:
            print_warning(f"Could not auto-start agent: {e}")
            print("  Start manually with: claude-memory-agent start")
    else:
        print_warning("Skipping auto-start (Ollama not running)")
        print("  After installing Ollama, run: claude-memory-agent start")

    # Done!
    print_post_install_instructions(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
