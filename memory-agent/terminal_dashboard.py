#!/usr/bin/env python3
"""Live Terminal Dashboard for Claude Memory Agent.

A standalone real-time dashboard that polls the running server and displays
live statistics using Rich's Live display.

Usage:
    python terminal_dashboard.py              # Default: http://localhost:8102
    python terminal_dashboard.py --port 8103  # Custom port
    python terminal_dashboard.py --refresh 3  # Refresh every 3 seconds

Press Ctrl+C to exit.
"""
import sys
import time
import argparse
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
from rich.progress_bar import ProgressBar
from rich.spinner import Spinner
from rich.theme import Theme

THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error.style": "bold red",
    "success": "bold green",
    "hot": "bold red",
    "warm": "bold yellow",
    "cold": "bold blue",
    "header": "bold magenta",
    "muted": "dim white",
    "accent": "bold cyan",
})

console = Console(theme=THEME)



class DashboardState:
    """Tracks dashboard state and history."""

    def __init__(self):
        self.health: Dict = {}
        self.stats: Dict = {}
        self.tier_stats: Dict = {}
        self.consolidation_stats: Dict = {}
        self.pipeline_stats: Dict = {}
        self.index_stats: Dict = {}
        self.decay_stats: Dict = {}
        self.recent_activity: list = []
        self.last_update: Optional[datetime] = None
        self.error: Optional[str] = None
        self.uptime_start: Optional[datetime] = None
        self.refresh_count: int = 0
        self.connection_errors: int = 0


def fetch_data(base_url: str, state: DashboardState):
    """Fetch all dashboard data from the server."""
    client = httpx.Client(timeout=5.0)
    state.error = None

    try:
        # Health check
        r = client.get(f"{base_url}/health")
        state.health = r.json()
        if state.uptime_start is None:
            state.uptime_start = datetime.now()

        # Stats
        try:
            r = client.get(f"{base_url}/api/stats")
            state.stats = r.json()
        except Exception:
            pass

        # Tier stats
        try:
            r = client.get(f"{base_url}/api/tiers/stats")
            state.tier_stats = r.json()
        except Exception:
            pass

        # Consolidation stats
        try:
            r = client.get(f"{base_url}/api/consolidation/stats")
            state.consolidation_stats = r.json()
        except Exception:
            pass

        # Embedding pipeline stats
        try:
            r = client.get(f"{base_url}/api/embedding-pipeline/stats")
            state.pipeline_stats = r.json()
        except Exception:
            pass

        # Index stats
        try:
            r = client.get(f"{base_url}/api/index-stats")
            state.index_stats = r.json()
        except Exception:
            pass

        # Decay stats
        try:
            r = client.get(f"{base_url}/api/decay/stats")
            state.decay_stats = r.json()
        except Exception:
            pass

        state.last_update = datetime.now()
        state.refresh_count += 1
        state.connection_errors = 0

    except httpx.ConnectError:
        state.error = "Cannot connect to server"
        state.connection_errors += 1
    except Exception as e:
        state.error = str(e)
        state.connection_errors += 1
    finally:
        client.close()


def make_header(state: DashboardState, base_url: str) -> Panel:
    """Create the header panel."""
    text = Text()
    text.append("  Claude", style="bold cyan")
    text.append("Memory", style="bold white")
    text.append("  ", style="")

    # Connection status
    if state.error:
        text.append(" DISCONNECTED ", style="bold white on red")
    else:
        text.append(" LIVE ", style="bold white on green")

    text.append(f"  {base_url}", style="muted")

    # Uptime
    if state.uptime_start:
        uptime = datetime.now() - state.uptime_start
        hours = int(uptime.total_seconds() // 3600)
        mins = int((uptime.total_seconds() % 3600) // 60)
        text.append(f"  |  Uptime: {hours}h{mins:02d}m", style="muted")

    text.append(f"  |  Refresh #{state.refresh_count}", style="muted")

    return Panel(Align.center(text), style="cyan", height=3)


def make_memory_panel(state: DashboardState) -> Panel:
    """Create the memory statistics panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("label", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    stats = state.stats
    total = stats.get('total_memories', 0)
    table.add_row("Total Memories", f"[bold]{total}[/bold]")

    # Type breakdown
    type_counts = stats.get('type_counts', {})
    if type_counts:
        for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            bar_width = min(int(count / max(total, 1) * 20), 20)
            bar = "[green]" + "=" * bar_width + "[/green]" + "[muted]" + "-" * (20 - bar_width) + "[/muted]"
            table.add_row(f"  {mtype}", f"{count:>4}  {bar}")

    table.add_row("", "")
    table.add_row("Patterns", str(stats.get('total_patterns', 0)))
    table.add_row("Timeline Events", str(stats.get('total_timeline_events', 0)))

    return Panel(table, title="[bold]Memories[/bold]", border_style="green")


def make_tier_panel(state: DashboardState) -> Panel:
    """Create the tier distribution panel."""
    table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    table.add_column("Tier", style="dim cyan", width=8)
    table.add_column("Count", style="bold white", width=8, justify="right")
    table.add_column("Avg Imp", style="dim white", width=8, justify="right")
    table.add_column("", width=20)

    tiers = state.tier_stats.get('tiers', {})
    total = state.tier_stats.get('total_memories', 1) or 1

    tier_styles = {'hot': 'hot', 'warm': 'warm', 'cold': 'cold'}

    for tier_name in ['hot', 'warm', 'cold']:
        info = tiers.get(tier_name, {'count': 0, 'avg_importance': 0})
        count = info.get('count', 0)
        avg_imp = info.get('avg_importance', 0)
        pct = count / total * 100

        bar_width = min(int(pct / 5), 20)
        style = tier_styles.get(tier_name, 'muted')
        bar = f"[{style}]" + "|" * bar_width + f"[/{style}]" + " " * (20 - bar_width)

        icon = {"hot": "[hot]***[/hot]", "warm": "[warm] ** [/warm]", "cold": "[cold]  * [/cold]"}.get(tier_name, "")
        table.add_row(
            f"[{style}]{tier_name.upper()}[/{style}] {icon}",
            str(count),
            f"{avg_imp:.1f}",
            f"{bar} {pct:.0f}%"
        )

    return Panel(table, title="[bold]Memory Tiers[/bold]", border_style="yellow")


def make_health_panel(state: DashboardState) -> Panel:
    """Create the system health panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("key", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    health = state.health

    # Server status
    status = health.get('status', 'unknown')
    if status == 'healthy':
        table.add_row("Server", "[green]Healthy[/green]")
    elif status == 'degraded':
        table.add_row("Server", "[yellow]Degraded[/yellow]")
    else:
        table.add_row("Server", f"[red]{status}[/red]")

    # Ollama
    ollama = health.get('ollama', {})
    if isinstance(ollama, dict):
        if ollama.get('healthy'):
            model = ollama.get('model', '?')
            table.add_row("Ollama", f"[green]OK[/green] ({model})")
        else:
            table.add_row("Ollama", "[red]Down[/red]")
    else:
        table.add_row("Ollama", str(ollama))

    # Database
    db_info = health.get('database', {})
    if isinstance(db_info, dict):
        if db_info.get('connected'):
            table.add_row("Database", "[green]Connected[/green]")
        else:
            table.add_row("Database", "[red]Disconnected[/red]")

    # Vector index
    index = health.get('vector_index', {})
    if isinstance(index, dict):
        faiss = "[green]FAISS[/green]" if index.get('faiss_available') else "[yellow]NumPy[/yellow]"
        table.add_row("Vector Index", faiss)

    return Panel(table, title="[bold]Health[/bold]", border_style="green")


def make_pipeline_panel(state: DashboardState) -> Panel:
    """Create the embedding pipeline panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("key", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    pipeline = state.pipeline_stats
    cache = pipeline.get('cache', {})

    if cache:
        size = cache.get('size', 0)
        max_size = cache.get('max_size', 0)
        hits = cache.get('hits', 0)
        misses = cache.get('misses', 0)
        hit_rate = cache.get('hit_rate', 0)
        mem_mb = cache.get('estimated_memory_mb', 0)

        # Cache fill bar
        fill_pct = size / max(max_size, 1) * 100
        fill_bar_w = min(int(fill_pct / 5), 20)
        fill_bar = "[cyan]" + "|" * fill_bar_w + "[/cyan]" + "[muted]" + "-" * (20 - fill_bar_w) + "[/muted]"

        table.add_row("Cache Fill", f"{size}/{max_size}  {fill_bar}")
        table.add_row("Hit Rate", f"[{'green' if hit_rate > 0.5 else 'yellow'}]{hit_rate:.1%}[/{'green' if hit_rate > 0.5 else 'yellow'}]")
        table.add_row("Hits / Misses", f"[green]{hits}[/green] / [yellow]{misses}[/yellow]")
        table.add_row("Memory", f"{mem_mb:.2f} MB")
    else:
        table.add_row("Status", "[muted]Not initialized[/muted]")

    precomputing = pipeline.get('precompute_running', False)
    if precomputing:
        table.add_row("Precompute", "[green]Running[/green]")
    else:
        table.add_row("Precompute", "[muted]Idle[/muted]")

    degraded = pipeline.get('service_degraded', False)
    if degraded:
        table.add_row("Ollama", "[red]Degraded[/red]")

    return Panel(table, title="[bold]Embedding Pipeline[/bold]", border_style="cyan")


def make_consolidation_panel(state: DashboardState) -> Panel:
    """Create the consolidation stats panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("key", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    cs = state.consolidation_stats

    consolidated = cs.get('consolidated_memories', 0)
    archived = cs.get('archived_originals', 0)
    avg_group = cs.get('avg_group_size', 0)

    table.add_row("Consolidated", str(consolidated))
    table.add_row("Archived", str(archived))
    table.add_row("Avg Group Size", str(avg_group))
    table.add_row("Space Saved", cs.get('space_savings_estimate', 'N/A'))

    return Panel(table, title="[bold]Consolidation[/bold]", border_style="magenta")


def make_decay_panel(state: DashboardState) -> Panel:
    """Create the memory decay stats panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("key", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    ds = state.decay_stats

    permanent = ds.get('permanent_count', 0)
    decayable = ds.get('decayable_count', 0)
    at_risk = ds.get('at_risk_count', 0)
    healthy = ds.get('healthy_count', 0)
    archived = ds.get('archived_by_decay', 0)

    table.add_row("Permanent", f"[green]{permanent}[/green]")
    table.add_row("Decayable", str(decayable))

    if decayable > 0:
        health_pct = healthy / max(decayable, 1) * 100
        risk_bar_w = min(int(at_risk / max(decayable, 1) * 20), 20)
        health_bar_w = 20 - risk_bar_w
        bar = "[green]" + "|" * health_bar_w + "[/green]" + "[red]" + "|" * risk_bar_w + "[/red]"
        table.add_row("Health", f"{bar} {health_pct:.0f}%")

    table.add_row("At Risk", f"[{'red' if at_risk > 0 else 'green'}]{at_risk}[/{'red' if at_risk > 0 else 'green'}]")
    table.add_row("Archived", str(archived))

    return Panel(table, title="[bold]Memory Decay[/bold]", border_style="yellow")


def make_index_panel(state: DashboardState) -> Panel:
    """Create the vector index stats panel."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("key", style="dim cyan", width=16)
    table.add_column("value", style="bold white")

    ix = state.index_stats

    faiss = ix.get('faiss_available', False)
    table.add_row("Backend", "[green]FAISS[/green]" if faiss else "[yellow]NumPy[/yellow]")

    for idx_name in ['memories', 'patterns', 'timeline']:
        idx = ix.get(idx_name, {})
        if idx:
            size = idx.get('size', 0)
            searches = idx.get('search_count', 0)
            table.add_row(f"  {idx_name}", f"{size} vectors, {searches} searches")

    return Panel(table, title="[bold]Vector Index[/bold]", border_style="blue")


def make_error_panel(state: DashboardState) -> Panel:
    """Create an error panel when server is unreachable."""
    text = Text()
    text.append("\n  Cannot connect to server\n\n", style="bold red")
    text.append(f"  {state.error}\n\n", style="muted")
    text.append(f"  Consecutive errors: {state.connection_errors}\n", style="yellow")
    text.append("  Retrying...\n", style="muted")
    return Panel(text, title="[bold red]Connection Error[/bold red]", border_style="red")


def build_layout(state: DashboardState, base_url: str) -> Layout:
    """Build the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # Header
    layout["header"].update(make_header(state, base_url))

    if state.error and state.connection_errors > 2:
        layout["body"].update(make_error_panel(state))
    else:
        # Body: two rows of panels
        layout["body"].split_column(
            Layout(name="top_row", size=14),
            Layout(name="bottom_row"),
        )

        layout["top_row"].split_row(
            Layout(make_memory_panel(state), name="memories"),
            Layout(make_tier_panel(state), name="tiers"),
            Layout(make_health_panel(state), name="health"),
        )

        layout["bottom_row"].split_row(
            Layout(make_pipeline_panel(state), name="pipeline"),
            Layout(make_consolidation_panel(state), name="consolidation"),
            Layout(make_decay_panel(state), name="decay"),
            Layout(make_index_panel(state), name="index"),
        )

    # Footer
    footer_text = Text()
    footer_text.append("  Press ", style="muted")
    footer_text.append("Ctrl+C", style="bold")
    footer_text.append(" to exit", style="muted")
    if state.last_update:
        footer_text.append(f"  |  Last updated: {state.last_update.strftime('%H:%M:%S')}", style="muted")
    layout["footer"].update(Panel(footer_text, style="muted"))

    return layout


def main():
    parser = argparse.ArgumentParser(description="Claude Memory Agent - Live Dashboard")
    parser.add_argument("--port", type=int, default=8102, help="Server port (default: 8102)")
    parser.add_argument("--host", type=str, default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds (default: 2)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    state = DashboardState()

    console.print(f"\n[bold cyan]Connecting to {base_url}...[/bold cyan]\n")

    # Initial fetch
    fetch_data(base_url, state)

    if state.error:
        console.print(f"[red]Warning: {state.error}[/red]")
        console.print("[muted]Dashboard will retry automatically...[/muted]\n")

    try:
        with Live(
            build_layout(state, base_url),
            console=console,
            refresh_per_second=1,
            screen=True,
        ) as live:
            while True:
                time.sleep(args.refresh)
                fetch_data(base_url, state)
                live.update(build_layout(state, base_url))

    except KeyboardInterrupt:
        console.print("\n[muted]Dashboard stopped.[/muted]")


if __name__ == "__main__":
    main()
