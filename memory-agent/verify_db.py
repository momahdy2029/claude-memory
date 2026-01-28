#!/usr/bin/env python3
"""Quick database verification tool - run this to see what's stored."""
import sqlite3
import json
from datetime import datetime

DB_PATH = "memories.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "="*60)
    print("   CLAUDE MEMORY - DATABASE VERIFICATION")
    print("="*60)

    # Agent configs
    print("\n[AGENT CONFIGURATIONS]")
    print("-" * 40)
    cursor = conn.execute("""
        SELECT project_path, agent_id, enabled, updated_at
        FROM project_agent_config
        ORDER BY updated_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            project = row["project_path"].split("\\")[-1] if "\\" in row["project_path"] else row["project_path"].split("/")[-1]
            status = "ON" if row["enabled"] else "OFF"
            print(f"  [{status:3}] {row['agent_id']:<30} | {project}")
            print(f"        Updated: {row['updated_at']}")
    else:
        print("  No configurations yet - toggle an agent in the dashboard!")

    # MCP configs
    print("\n[MCP CONFIGURATIONS]")
    print("-" * 40)
    cursor = conn.execute("""
        SELECT project_path, mcp_id, enabled, updated_at
        FROM project_mcp_config
        ORDER BY updated_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            project = row["project_path"].split("\\")[-1] if "\\" in row["project_path"] else row["project_path"].split("/")[-1]
            status = "ON" if row["enabled"] else "OFF"
            print(f"  [{status:3}] {row['mcp_id']:<30} | {project}")
    else:
        print("  No MCP configurations yet")

    # Hook configs
    print("\n[HOOK CONFIGURATIONS]")
    print("-" * 40)
    cursor = conn.execute("""
        SELECT project_path, hook_id, enabled, updated_at
        FROM project_hook_config
        ORDER BY updated_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            project = row["project_path"].split("\\")[-1] if "\\" in row["project_path"] else row["project_path"].split("/")[-1]
            status = "ON" if row["enabled"] else "OFF"
            print(f"  [{status:3}] {row['hook_id']:<30} | {project}")
    else:
        print("  No hook configurations yet")

    # Timeline events
    print("\n[RECENT TIMELINE EVENTS]")
    print("-" * 40)
    cursor = conn.execute("""
        SELECT event_type, summary, session_id, created_at, is_anchor
        FROM timeline_events
        ORDER BY created_at DESC
        LIMIT 15
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            anchor = " [ANCHOR]" if row["is_anchor"] else ""
            print(f"  [{row['event_type']:<12}] {row['summary'][:50]}...{anchor}")
            print(f"               {row['created_at']}")
    else:
        print("  No timeline events yet")

    # Anchors only
    print("\n[ANCHORS (VERIFIED FACTS)]")
    print("-" * 40)
    cursor = conn.execute("""
        SELECT summary, details, created_at
        FROM timeline_events
        WHERE is_anchor = 1
        ORDER BY created_at DESC
        LIMIT 10
    """)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"  * {row['summary']}")
            if row['details']:
                print(f"    Details: {row['details'][:60]}...")
    else:
        print("  No anchors set yet")

    # Stats
    print("\n[STATISTICS]")
    print("-" * 40)
    cursor = conn.execute("SELECT COUNT(*) as c FROM memories")
    print(f"  Total Memories: {cursor.fetchone()['c']}")

    cursor = conn.execute("SELECT COUNT(*) as c FROM timeline_events")
    print(f"  Timeline Events: {cursor.fetchone()['c']}")

    cursor = conn.execute("SELECT COUNT(*) as c FROM project_agent_config")
    print(f"  Agent Configs: {cursor.fetchone()['c']}")

    cursor = conn.execute("SELECT COUNT(*) as c FROM session_state")
    print(f"  Sessions: {cursor.fetchone()['c']}")

    cursor = conn.execute("SELECT COUNT(*) as c FROM checkpoints")
    print(f"  Checkpoints: {cursor.fetchone()['c']}")

    print("\n" + "="*60)
    print("  Run this again after toggling something in the dashboard!")
    print("="*60 + "\n")

    conn.close()

if __name__ == "__main__":
    main()
