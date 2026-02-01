"""Context tagging skill for context-aware memory ranking.

Context-aware memory system that tracks where solutions worked or failed.
Same solution may work in React but fail in Vue - context tags capture this.

Context structure:
{
    "project_type": "react" | "python" | "wordpress" | "vue" | etc.,
    "tech_stack": ["typescript", "fastapi", "docker"],
    "environment": "dev" | "prod" | "test",
    "file_patterns": ["*.tsx", "*.py", "*.php"]
}
"""
import json
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


def detect_project_context(project_path: Optional[str]) -> Dict[str, Any]:
    """Auto-detect context from project path by analyzing files and configs.

    Args:
        project_path: Path to the project directory

    Returns:
        Context dict with project_type, tech_stack, environment, file_patterns
    """
    if not project_path or not os.path.exists(project_path):
        return {}

    context = {
        "project_type": None,
        "tech_stack": [],
        "environment": "dev",  # Default to dev
        "file_patterns": []
    }

    path = Path(project_path)

    # Detect based on config files
    config_detectors = {
        "package.json": _detect_js_context,
        "requirements.txt": _detect_python_context,
        "pyproject.toml": _detect_python_context,
        "Pipfile": _detect_python_context,
        "composer.json": _detect_php_context,
        "wp-config.php": _detect_wordpress_context,
        "Cargo.toml": _detect_rust_context,
        "go.mod": _detect_go_context,
        "pom.xml": _detect_java_context,
        "build.gradle": _detect_java_context,
    }

    for config_file, detector in config_detectors.items():
        config_path = path / config_file
        if config_path.exists():
            try:
                detected = detector(config_path)
                _merge_context(context, detected)
            except Exception:
                pass  # Ignore parsing errors

    # Fallback: detect from directory structure
    if not context["project_type"]:
        context["project_type"] = _detect_from_structure(path)

    # Detect file patterns present
    context["file_patterns"] = _detect_file_patterns(path)

    return context


def _detect_js_context(package_json_path: Path) -> Dict[str, Any]:
    """Detect JavaScript/TypeScript project context."""
    context = {"tech_stack": []}

    try:
        with open(package_json_path, 'r', encoding='utf-8') as f:
            pkg = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"project_type": "nodejs"}

    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}

    # Detect framework
    if "react" in deps:
        context["project_type"] = "react"
        context["tech_stack"].append("react")
        if "next" in deps:
            context["project_type"] = "nextjs"
            context["tech_stack"].append("nextjs")
    elif "vue" in deps:
        context["project_type"] = "vue"
        context["tech_stack"].append("vue")
        if "nuxt" in deps:
            context["project_type"] = "nuxt"
            context["tech_stack"].append("nuxt")
    elif "angular" in deps or "@angular/core" in deps:
        context["project_type"] = "angular"
        context["tech_stack"].append("angular")
    elif "svelte" in deps:
        context["project_type"] = "svelte"
        context["tech_stack"].append("svelte")
    elif "express" in deps:
        context["project_type"] = "express"
        context["tech_stack"].append("express")
    else:
        context["project_type"] = "nodejs"

    # Detect TypeScript
    if "typescript" in deps:
        context["tech_stack"].append("typescript")

    # Detect testing frameworks
    if "jest" in deps:
        context["tech_stack"].append("jest")
    if "vitest" in deps:
        context["tech_stack"].append("vitest")
    if "cypress" in deps:
        context["tech_stack"].append("cypress")

    # Detect styling
    if "tailwindcss" in deps:
        context["tech_stack"].append("tailwind")
    if "styled-components" in deps:
        context["tech_stack"].append("styled-components")

    return context


def _detect_python_context(config_path: Path) -> Dict[str, Any]:
    """Detect Python project context."""
    context = {"project_type": "python", "tech_stack": ["python"]}

    try:
        content = config_path.read_text(encoding='utf-8')
    except IOError:
        return context

    content_lower = content.lower()

    # Detect frameworks
    if "django" in content_lower:
        context["project_type"] = "django"
        context["tech_stack"].append("django")
    elif "flask" in content_lower:
        context["project_type"] = "flask"
        context["tech_stack"].append("flask")
    elif "fastapi" in content_lower:
        context["project_type"] = "fastapi"
        context["tech_stack"].append("fastapi")

    # Detect other tools
    if "pytest" in content_lower:
        context["tech_stack"].append("pytest")
    if "celery" in content_lower:
        context["tech_stack"].append("celery")
    if "sqlalchemy" in content_lower:
        context["tech_stack"].append("sqlalchemy")
    if "pandas" in content_lower:
        context["tech_stack"].append("pandas")

    return context


def _detect_php_context(composer_path: Path) -> Dict[str, Any]:
    """Detect PHP project context."""
    context = {"project_type": "php", "tech_stack": ["php"]}

    try:
        with open(composer_path, 'r', encoding='utf-8') as f:
            composer = json.load(f)
    except (json.JSONDecodeError, IOError):
        return context

    require = {**composer.get("require", {}), **composer.get("require-dev", {})}

    if "laravel/framework" in require:
        context["project_type"] = "laravel"
        context["tech_stack"].append("laravel")
    elif "symfony/framework-bundle" in require:
        context["project_type"] = "symfony"
        context["tech_stack"].append("symfony")

    return context


def _detect_wordpress_context(wp_config_path: Path) -> Dict[str, Any]:
    """Detect WordPress context."""
    context = {
        "project_type": "wordpress",
        "tech_stack": ["php", "wordpress", "mysql"]
    }

    # Check if this is a plugin or theme
    parent = wp_config_path.parent
    if (parent / "wp-content" / "plugins").exists():
        # Could be a plugin development environment
        pass
    if (parent / "wp-content" / "themes").exists():
        # Could be a theme development environment
        pass

    return context


def _detect_rust_context(cargo_path: Path) -> Dict[str, Any]:
    """Detect Rust project context."""
    return {"project_type": "rust", "tech_stack": ["rust"]}


def _detect_go_context(go_mod_path: Path) -> Dict[str, Any]:
    """Detect Go project context."""
    return {"project_type": "go", "tech_stack": ["go"]}


def _detect_java_context(config_path: Path) -> Dict[str, Any]:
    """Detect Java project context."""
    context = {"project_type": "java", "tech_stack": ["java"]}

    try:
        content = config_path.read_text(encoding='utf-8')
        if "spring" in content.lower():
            context["project_type"] = "spring"
            context["tech_stack"].append("spring")
    except IOError:
        pass

    return context


def _detect_from_structure(path: Path) -> Optional[str]:
    """Fallback detection from directory structure."""
    # Check for common directory patterns
    if (path / "src" / "main" / "java").exists():
        return "java"
    if (path / "src" / "components").exists():
        return "react"
    if (path / "app" / "Http" / "Controllers").exists():
        return "laravel"
    if (path / "manage.py").exists():
        return "django"

    return None


def _detect_file_patterns(path: Path) -> List[str]:
    """Detect common file patterns in the project."""
    patterns = set()
    extensions_found = set()

    # Sample files (limit depth and count for performance)
    try:
        for item in path.rglob("*"):
            if item.is_file() and not any(p in str(item) for p in ['node_modules', '.git', 'venv', '__pycache__']):
                ext = item.suffix
                if ext:
                    extensions_found.add(ext)
            if len(extensions_found) > 20:
                break
    except (PermissionError, OSError):
        pass

    # Map to glob patterns
    extension_map = {
        ".tsx": "*.tsx",
        ".ts": "*.ts",
        ".jsx": "*.jsx",
        ".js": "*.js",
        ".py": "*.py",
        ".php": "*.php",
        ".vue": "*.vue",
        ".svelte": "*.svelte",
        ".rs": "*.rs",
        ".go": "*.go",
        ".java": "*.java",
        ".rb": "*.rb",
    }

    for ext in extensions_found:
        if ext in extension_map:
            patterns.add(extension_map[ext])

    return list(patterns)[:10]  # Limit patterns


def _merge_context(target: Dict, source: Dict):
    """Merge source context into target."""
    if source.get("project_type"):
        target["project_type"] = source["project_type"]

    if source.get("tech_stack"):
        existing = set(target.get("tech_stack", []))
        existing.update(source["tech_stack"])
        target["tech_stack"] = list(existing)

    if source.get("environment"):
        target["environment"] = source["environment"]


def calculate_context_similarity(context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
    """Calculate similarity between two contexts (0.0 to 1.0).

    Scoring:
    - project_type match: 0.4 weight
    - tech_stack overlap: 0.4 weight (Jaccard similarity)
    - file_patterns overlap: 0.2 weight (Jaccard similarity)
    """
    if not context1 or not context2:
        return 0.0

    score = 0.0

    # Project type match (0.4 weight)
    type1 = context1.get("project_type", "").lower()
    type2 = context2.get("project_type", "").lower()
    if type1 and type2:
        if type1 == type2:
            score += 0.4
        # Partial matches for related types
        elif _are_related_types(type1, type2):
            score += 0.2

    # Tech stack overlap (0.4 weight) - Jaccard similarity
    stack1 = set(s.lower() for s in context1.get("tech_stack", []))
    stack2 = set(s.lower() for s in context2.get("tech_stack", []))
    if stack1 or stack2:
        intersection = len(stack1 & stack2)
        union = len(stack1 | stack2)
        if union > 0:
            score += 0.4 * (intersection / union)

    # File patterns overlap (0.2 weight) - Jaccard similarity
    patterns1 = set(context1.get("file_patterns", []))
    patterns2 = set(context2.get("file_patterns", []))
    if patterns1 or patterns2:
        intersection = len(patterns1 & patterns2)
        union = len(patterns1 | patterns2)
        if union > 0:
            score += 0.2 * (intersection / union)

    return round(score, 4)


def _are_related_types(type1: str, type2: str) -> bool:
    """Check if two project types are related."""
    related_groups = [
        {"react", "nextjs", "gatsby"},
        {"vue", "nuxt"},
        {"python", "django", "flask", "fastapi"},
        {"php", "laravel", "symfony", "wordpress"},
        {"nodejs", "express"},
        {"java", "spring"},
    ]

    for group in related_groups:
        if type1 in group and type2 in group:
            return True

    return False


async def add_context_success(
    db,
    memory_id: int,
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Record that a memory's solution worked in a specific context.

    Args:
        db: Database service instance
        memory_id: ID of the memory
        context: Context where the solution worked

    Returns:
        Dict with success status and updated context info
    """
    cursor = db.conn.cursor()

    # Get current worked_in contexts
    cursor.execute(
        "SELECT worked_in, failed_in, context_confidence FROM memories WHERE id = ?",
        [memory_id]
    )
    row = cursor.fetchone()

    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found"
        }

    # Parse existing contexts
    worked_in = json.loads(row["worked_in"]) if row["worked_in"] else []
    failed_in = json.loads(row["failed_in"]) if row["failed_in"] else []
    current_confidence = row["context_confidence"] if row["context_confidence"] is not None else 0.5

    # Add new context to worked_in (avoid duplicates)
    context_key = _context_to_key(context)
    existing_keys = [_context_to_key(c) for c in worked_in]

    if context_key not in existing_keys:
        worked_in.append(context)

    # Recalculate context confidence
    # More worked_in contexts = higher confidence
    # Any failed_in contexts = lower confidence
    success_count = len(worked_in)
    failure_count = len(failed_in)
    total = success_count + failure_count

    if total > 0:
        new_confidence = success_count / total
        # Apply smoothing for low sample sizes
        new_confidence = (new_confidence * total + 0.5) / (total + 1)
    else:
        new_confidence = 0.5

    # Update database
    cursor.execute(
        """
        UPDATE memories
        SET worked_in = ?,
            context_confidence = ?,
            updated_at = datetime('now')
        WHERE id = ?
        """,
        [json.dumps(worked_in), new_confidence, memory_id]
    )
    db.conn.commit()

    return {
        "success": True,
        "memory_id": memory_id,
        "context_added": context,
        "worked_in_count": len(worked_in),
        "failed_in_count": len(failed_in),
        "old_confidence": current_confidence,
        "new_confidence": new_confidence,
        "message": f"Context success recorded. Confidence: {current_confidence:.3f} -> {new_confidence:.3f}"
    }


async def add_context_failure(
    db,
    memory_id: int,
    context: Dict[str, Any],
    failure_reason: Optional[str] = None
) -> Dict[str, Any]:
    """Record that a memory's solution failed in a specific context.

    Args:
        db: Database service instance
        memory_id: ID of the memory
        context: Context where the solution failed
        failure_reason: Optional explanation of why it failed

    Returns:
        Dict with success status and updated context info
    """
    cursor = db.conn.cursor()

    # Get current contexts
    cursor.execute(
        "SELECT worked_in, failed_in, context_confidence FROM memories WHERE id = ?",
        [memory_id]
    )
    row = cursor.fetchone()

    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found"
        }

    # Parse existing contexts
    worked_in = json.loads(row["worked_in"]) if row["worked_in"] else []
    failed_in = json.loads(row["failed_in"]) if row["failed_in"] else []
    current_confidence = row["context_confidence"] if row["context_confidence"] is not None else 0.5

    # Add failure reason to context if provided
    if failure_reason:
        context["failure_reason"] = failure_reason

    # Add new context to failed_in (avoid duplicates)
    context_key = _context_to_key(context)
    existing_keys = [_context_to_key(c) for c in failed_in]

    if context_key not in existing_keys:
        failed_in.append(context)

    # Recalculate context confidence
    success_count = len(worked_in)
    failure_count = len(failed_in)
    total = success_count + failure_count

    if total > 0:
        new_confidence = success_count / total
        new_confidence = (new_confidence * total + 0.5) / (total + 1)
    else:
        new_confidence = 0.5

    # Update database
    cursor.execute(
        """
        UPDATE memories
        SET failed_in = ?,
            context_confidence = ?,
            updated_at = datetime('now')
        WHERE id = ?
        """,
        [json.dumps(failed_in), new_confidence, memory_id]
    )
    db.conn.commit()

    return {
        "success": True,
        "memory_id": memory_id,
        "context_added": context,
        "worked_in_count": len(worked_in),
        "failed_in_count": len(failed_in),
        "old_confidence": current_confidence,
        "new_confidence": new_confidence,
        "message": f"Context failure recorded. Confidence: {current_confidence:.3f} -> {new_confidence:.3f}"
    }


async def get_context_score(
    db,
    memory_id: int,
    current_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate how relevant a memory is for the current context.

    Scoring logic:
    - High similarity to worked_in contexts: boost score
    - High similarity to failed_in contexts: reduce score
    - No context data: neutral (0.0 adjustment)

    Args:
        db: Database service instance
        memory_id: ID of the memory
        current_context: Current project context

    Returns:
        Dict with context score (-0.2 to +0.2 adjustment)
    """
    cursor = db.conn.cursor()

    cursor.execute(
        "SELECT worked_in, failed_in, context_confidence FROM memories WHERE id = ?",
        [memory_id]
    )
    row = cursor.fetchone()

    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found"
        }

    worked_in = json.loads(row["worked_in"]) if row["worked_in"] else []
    failed_in = json.loads(row["failed_in"]) if row["failed_in"] else []
    context_confidence = row["context_confidence"] if row["context_confidence"] is not None else None

    # No context data - neutral
    if not worked_in and not failed_in:
        return {
            "success": True,
            "memory_id": memory_id,
            "context_score": 0.0,
            "context_adjustment": 0.0,
            "has_context_data": False,
            "context_confidence": context_confidence
        }

    # Calculate similarity to worked_in contexts
    max_success_similarity = 0.0
    for ctx in worked_in:
        sim = calculate_context_similarity(current_context, ctx)
        max_success_similarity = max(max_success_similarity, sim)

    # Calculate similarity to failed_in contexts
    max_failure_similarity = 0.0
    for ctx in failed_in:
        sim = calculate_context_similarity(current_context, ctx)
        max_failure_similarity = max(max_failure_similarity, sim)

    # Calculate adjustment (-0.2 to +0.2)
    # Boost if similar to success, penalty if similar to failure
    success_boost = max_success_similarity * 0.2
    failure_penalty = max_failure_similarity * 0.2
    context_adjustment = success_boost - failure_penalty

    # Determine recommendation
    recommendation = "neutral"
    if context_adjustment > 0.1:
        recommendation = "recommended"
    elif context_adjustment < -0.1:
        recommendation = "caution"

    return {
        "success": True,
        "memory_id": memory_id,
        "context_score": round(max_success_similarity - max_failure_similarity, 4),
        "context_adjustment": round(context_adjustment, 4),
        "has_context_data": True,
        "worked_in_similarity": round(max_success_similarity, 4),
        "failed_in_similarity": round(max_failure_similarity, 4),
        "worked_in_count": len(worked_in),
        "failed_in_count": len(failed_in),
        "context_confidence": context_confidence,
        "recommendation": recommendation
    }


async def get_memory_contexts(
    db,
    memory_id: int
) -> Dict[str, Any]:
    """Get all context data for a memory.

    Args:
        db: Database service instance
        memory_id: ID of the memory

    Returns:
        Dict with worked_in, failed_in, and context confidence
    """
    cursor = db.conn.cursor()

    cursor.execute(
        """
        SELECT id, content, worked_in, failed_in, context_confidence,
               project_type, tech_stack
        FROM memories WHERE id = ?
        """,
        [memory_id]
    )
    row = cursor.fetchone()

    if not row:
        return {
            "success": False,
            "error": f"Memory with ID {memory_id} not found"
        }

    worked_in = json.loads(row["worked_in"]) if row["worked_in"] else []
    failed_in = json.loads(row["failed_in"]) if row["failed_in"] else []
    tech_stack = json.loads(row["tech_stack"]) if row["tech_stack"] else []

    return {
        "success": True,
        "memory_id": memory_id,
        "content_preview": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
        "original_context": {
            "project_type": row["project_type"],
            "tech_stack": tech_stack
        },
        "worked_in": worked_in,
        "failed_in": failed_in,
        "context_confidence": row["context_confidence"],
        "worked_in_count": len(worked_in),
        "failed_in_count": len(failed_in)
    }


def _context_to_key(context: Dict[str, Any]) -> str:
    """Create a unique key for a context (for deduplication)."""
    # Use project_type + sorted tech_stack as key
    project_type = context.get("project_type", "")
    tech_stack = sorted(context.get("tech_stack", []))
    return f"{project_type}:{','.join(tech_stack)}"
