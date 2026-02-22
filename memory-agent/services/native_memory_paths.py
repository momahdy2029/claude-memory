"""Native Memory Paths - Maps project paths to Claude Code's auto memory directories.

Claude Code stores per-project auto memory at:
    ~/.claude/projects/<slug>/memory/MEMORY.md

The slug is derived from the project's absolute path:
    C:\\xampp\\htdocs\\server  ->  C--xampp-htdocs-server
    D:\\Desktop-Projects\\foo  ->  D--Desktop-Projects-foo

Algorithm (reverse-engineered from existing directories):
    - Drive colon+separator (:\\ or :/) becomes --
    - All remaining separators (\\ or /) become -
    - Spaces become -
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"


def project_path_to_slug(project_path: str) -> str:
    """Convert an absolute project path to Claude Code's slug format.

    Examples:
        C:\\xampp\\htdocs\\server      -> C--xampp-htdocs-server
        D:\\Desktop-Projects\\foo      -> D--Desktop-Projects-foo
        C:\\Users\\moham\\Desktop\\X   -> C--Users-moham-Desktop-X
    """
    # Normalize to forward slashes
    p = project_path.replace("\\", "/").rstrip("/")

    # Handle drive letter: C:/ -> C--
    if len(p) >= 2 and p[1] == ":":
        drive = p[0]
        rest = p[2:].lstrip("/")
        slug = f"{drive}--{rest}"
    else:
        slug = p.lstrip("/")

    # Replace remaining separators and spaces with -
    slug = slug.replace("/", "-").replace(" ", "-")

    return slug


def get_native_memory_dir(project_path: str) -> Path:
    """Return the native auto memory directory for a project.

    Returns ~/.claude/projects/<slug>/memory/
    """
    slug = project_path_to_slug(project_path)
    return PROJECTS_DIR / slug / "memory"


def get_native_memory_md(project_path: str) -> Path:
    """Return the path to the native MEMORY.md for a project."""
    return get_native_memory_dir(project_path) / "MEMORY.md"


def list_native_memory_files(project_path: str) -> List[Path]:
    """List all markdown files in a project's native memory directory.

    Returns MEMORY.md first (if exists), then topic files alphabetically.
    """
    mem_dir = get_native_memory_dir(project_path)
    if not mem_dir.exists():
        return []

    memory_md = mem_dir / "MEMORY.md"
    files = []

    if memory_md.exists():
        files.append(memory_md)

    # Add topic files (any .md that isn't MEMORY.md)
    for f in sorted(mem_dir.glob("*.md")):
        if f.name != "MEMORY.md":
            files.append(f)

    return files


