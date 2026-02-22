"""Rich Terminal UI for Claude Memory Agent.

Provides:
- Colorful startup splash screen with system stats
- Rich logging handler with colored levels
- Helper panels for status display
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.logging import RichHandler
from rich.theme import Theme
from rich.align import Align

# Custom theme for the memory agent
MEMORY_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "memory.hot": "bold red",
    "memory.warm": "bold yellow",
    "memory.cold": "bold blue",
    "header": "bold magenta",
    "muted": "dim white",
    "accent": "bold cyan",
    "value": "bold white",
})

console = Console(theme=MEMORY_THEME)

# ASCII art logo
LOGO_MINI = r"""
  ___ _              _     __  __
 / __| |__ _ _  _ __| |___|  \/  |___ _ __  ___ _ _ _  _
| (__| / _` | || / _` / -_) |\/| / -_) '  \/ _ \ '_| || |
 \___|_\__,_|\_,_\__,_\___|_|  |_\___|_|_|_\___/_|  \_, |
                                                     |__/  """


def print_splash(
    version: str = "2.4.0",
    port: int = 8102,
    auth_enabled: bool = False,
    auth_keys: int = 0,
    queue_depth: int = 0,
    curator_interval: int = 24,
    embedding_cache_size: int = 500,
    precompute_interval: int = 60,
    consolidation_threshold: float = 0.85,
    consolidation_interval: int = 12,
    db_stats: Optional[Dict[str, Any]] = None,
):
    """Print the startup splash screen with system info.

    Args:
        version: Server version
        port: Server port
        auth_enabled: Whether auth is enabled
        auth_keys: Number of active auth keys
        queue_depth: Retry queue depth
        curator_interval: Curator check interval in hours
        embedding_cache_size: LRU cache size
        precompute_interval: Precompute loop interval in seconds
        consolidation_threshold: Similarity threshold for consolidation
        consolidation_interval: Consolidation loop interval in hours
        db_stats: Optional database statistics dict
    """
    console.print()

    # Logo panel
    logo_text = Text(LOGO_MINI, style="bold cyan")
    console.print(Panel(
        Align.center(logo_text),
        border_style="cyan",
        padding=(0, 2),
    ))

    # Version bar
    version_text = Text()
    version_text.append("  v", style="muted")
    version_text.append(version, style="bold green")
    version_text.append("  (CLaRa)  ", style="muted")
    version_text.append("|", style="muted")
    version_text.append(f"  Port {port}  ", style="accent")
    version_text.append("|", style="muted")
    version_text.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  ", style="muted")
    console.print(Align.center(version_text))
    console.print()

    # Status panels in columns
    panels = []

    # Server panel
    server_table = Table(show_header=False, box=None, padding=(0, 1))
    server_table.add_column("key", style="muted", width=18)
    server_table.add_column("value", style="value")

    auth_status = "[green]ENABLED[/green]" if auth_enabled else "[yellow]DISABLED[/yellow]"
    server_table.add_row("Authentication", auth_status)
    if auth_enabled:
        server_table.add_row("Active Keys", str(auth_keys))
    server_table.add_row("Retry Queue", f"{queue_depth} pending")
    server_table.add_row("Curator", f"every {curator_interval}h")

    panels.append(Panel(server_table, title="[bold]Server[/bold]", border_style="green", width=36))

    # CLaRa Features panel
    clara_table = Table(show_header=False, box=None, padding=(0, 1))
    clara_table.add_column("key", style="muted", width=18)
    clara_table.add_column("value", style="value")

    clara_table.add_row("Embedding Cache", f"{embedding_cache_size} slots")
    clara_table.add_row("Precompute", f"every {precompute_interval}s")
    clara_table.add_row("Consolidation", f"every {consolidation_interval}h")
    clara_table.add_row("Similarity", f">= {consolidation_threshold}")

    panels.append(Panel(clara_table, title="[bold]CLaRa Engine[/bold]", border_style="cyan", width=36))

    # Database panel (if stats available)
    if db_stats:
        db_table = Table(show_header=False, box=None, padding=(0, 1))
        db_table.add_column("key", style="muted", width=18)
        db_table.add_column("value", style="value")

        total = db_stats.get('total_memories', '?')
        db_table.add_row("Memories", f"[bold]{total}[/bold]")

        # Tier breakdown if available
        tier_counts = db_stats.get('tier_counts', {})
        if tier_counts:
            hot = tier_counts.get('hot', 0)
            warm = tier_counts.get('warm', 0)
            cold = tier_counts.get('cold', 0)
            db_table.add_row("Tiers", f"[memory.hot]{hot}[/memory.hot] / [memory.warm]{warm}[/memory.warm] / [memory.cold]{cold}[/memory.cold]")
        else:
            db_table.add_row("Tiers", "[muted]not evaluated[/muted]")

        patterns = db_stats.get('total_patterns', '?')
        db_table.add_row("Patterns", str(patterns))

        panels.append(Panel(db_table, title="[bold]Database[/bold]", border_style="yellow", width=36))

    console.print(Columns(panels, align="center", padding=(0, 1)))
    console.print()

    # Ready message
    console.print(
        Align.center(
            Text.assemble(
                ("  Ready  ", "bold white on green"),
                ("  ", ""),
                (f"http://localhost:{port}", "bold underline cyan"),
                ("  |  ", "muted"),
                (f"Dashboard: http://localhost:{port}/dashboard", "cyan"),
            )
        )
    )
    console.print()
    _print_separator()


def _print_separator():
    """Print a subtle separator line."""
    console.print("[muted]" + "-" * 70 + "[/muted]", justify="center")
    console.print()


def setup_rich_logging(level: str = "INFO") -> logging.Handler:
    """Set up rich logging handler for the application.

    Returns:
        RichHandler instance configured for the memory agent
    """
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        log_time_format="[%H:%M:%S]",
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Format
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    return handler


