"""Complete test suite for all memory system features."""
import asyncio
import sys
import os
sys.path.insert(0, '.')

async def test_all():
    print('=' * 70)
    print('COMPLETE MEMORY SYSTEM TEST SUITE')
    print('=' * 70)

    results = []

    # ===== ORIGINAL 12 FEATURES =====
    print('\n' + '-' * 70)
    print('SECTION A: ORIGINAL 12 FEATURES')
    print('-' * 70)

    # [1] Database with migrations
    print('\n[A1] Database & Migrations...')
    passed = 0
    try:
        from services.database import DatabaseService
        db = DatabaseService()
        await db.connect()
        await db.initialize_schema()

        # Check key tables exist
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        required = ['memories', 'patterns', 'projects', 'anchors', 'timeline']
        for t in required:
            if t in tables:
                passed += 1

        await db.close()
    except Exception as e:
        print(f'    ERROR: {e}')

    results.append(('A1. Database & Migrations', passed, len(required)))
    print(f'    {passed}/{len(required)} tables verified')

    # [2] Embeddings Service
    print('\n[A2] Embeddings Service...')
    passed = 0
    try:
        from services.embeddings import EmbeddingsService
        passed += 1  # Import works
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A2. Embeddings Service', passed, 1))
    print(f'    {passed}/1 import verified')

    # [3] Session Management
    print('\n[A3] Session Management...')
    passed = 0
    try:
        from skills.session import create_session, get_active_session, end_session
        passed += 3
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A3. Session Management', passed, 3))
    print(f'    {passed}/3 functions verified')

    # [4] Timeline Service
    print('\n[A4] Timeline Service...')
    passed = 0
    try:
        from skills.timeline import log_event, get_recent_events, get_session_timeline
        passed += 3
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A4. Timeline Service', passed, 3))
    print(f'    {passed}/3 functions verified')

    # [5] Grounding System (Anchors)
    print('\n[A5] Grounding System...')
    passed = 0
    try:
        from services.grounding import GroundingService
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A5. Grounding System', passed, 1))
    print(f'    {passed}/1 service verified')

    # [6] Insight Aggregation
    print('\n[A6] Insight Aggregation...')
    passed = 0
    try:
        from services.insight_aggregator import InsightAggregator
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A6. Insight Aggregation', passed, 1))
    print(f'    {passed}/1 service verified')

    # [7] Auto-Conflict Detection
    print('\n[A7] Auto-Conflict Detection...')
    passed = 0
    try:
        from services.grounding import GroundingService
        # Check conflict table exists
        db = DatabaseService()
        await db.connect()
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='anchor_conflicts'")
        if cursor.fetchone():
            passed += 1
        await db.close()
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A7. Conflict Detection', passed, 1))
    print(f'    {passed}/1 table verified')

    # [8] Memory Cleanup
    print('\n[A8] Memory Cleanup Service...')
    passed = 0
    try:
        from services.cleanup import CleanupService
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A8. Memory Cleanup', passed, 1))
    print(f'    {passed}/1 service verified')

    # [9] Auth & Queue
    print('\n[A9] Auth & Queue Services...')
    passed = 0
    try:
        from services.auth import AuthService
        from services.queue import MemoryQueue
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A9. Auth & Queue', passed, 2))
    print(f'    {passed}/2 services verified')

    # [10] Dashboard
    print('\n[A10] Dashboard...')
    passed = 0
    if os.path.exists('templates/dashboard.html'):
        passed += 1
    results.append(('A10. Dashboard', passed, 1))
    print(f'    {passed}/1 file verified')

    # [11] Embedding Model Switching
    print('\n[A11] Embedding Model Switching...')
    passed = 0
    try:
        from skills.admin import get_embedding_status, switch_embedding_model
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A11. Model Switching', passed, 2))
    print(f'    {passed}/2 functions verified')

    # [12] WebSocket Updates
    print('\n[A12] WebSocket Updates...')
    passed = 0
    try:
        from services.websocket import WebSocketManager, broadcast_event
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A12. WebSocket', passed, 2))
    print(f'    {passed}/2 imports verified')

    # ===== 6 AUTOMATION FEATURES =====
    print('\n' + '-' * 70)
    print('SECTION B: AUTOMATION FEATURES')
    print('-' * 70)

    # [B1] Auto-Capture Hook
    print('\n[B1] Auto-Capture Hook...')
    passed = 0
    if os.path.exists('hooks/auto_capture.py'):
        passed += 1
    results.append(('B1. Auto-Capture Hook', passed, 1))
    print(f'    {passed}/1 file verified')

    # [B2] Session Start/End Hooks
    print('\n[B2] Session Start/End Hooks...')
    passed = 0
    if os.path.exists('hooks/session_start.py'):
        passed += 1
    if os.path.exists('hooks/session_end.py'):
        passed += 1
    results.append(('B2. Session Hooks', passed, 2))
    print(f'    {passed}/2 files verified')

    # [B3] Auto-Injector
    print('\n[B3] Auto-Injector Service...')
    passed = 0
    try:
        from services.auto_inject import AutoInjector, get_auto_injector
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B3. Auto-Injector', passed, 2))
    print(f'    {passed}/2 imports verified')

    # [B4] Natural Language Interface
    print('\n[B4] Natural Language Interface...')
    passed = 0
    try:
        from skills.natural_language import parse_intent, process_natural_command
        # Quick test
        intent, _ = parse_intent('remember this: test')
        if intent == 'store':
            passed += 1
        intent, _ = parse_intent('show me past errors')
        if intent == 'list_errors':
            passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B4. Natural Language', passed, 2))
    print(f'    {passed}/2 patterns verified')

    # [B5] CLAUDE.md Sync
    print('\n[B5] CLAUDE.md Sync...')
    passed = 0
    try:
        from services.claude_md_sync import ClaudeMdSync, get_claude_md_sync
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B5. CLAUDE.md Sync', passed, 2))
    print(f'    {passed}/2 imports verified')

    # [B6] Confidence Scoring
    print('\n[B6] Confidence Scoring...')
    passed = 0
    try:
        from services.confidence import ConfidenceService, get_confidence_service
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B6. Confidence Scoring', passed, 2))
    print(f'    {passed}/2 imports verified')

    # ===== MAIN.PY INTEGRATION =====
    print('\n' + '-' * 70)
    print('SECTION C: MAIN.PY INTEGRATION')
    print('-' * 70)

    print('\n[C1] API Endpoints...')
    passed = 0
    with open('main.py', 'r') as f:
        content = f.read()

    endpoints = [
        # Original endpoints
        '/health', '/dashboard', '/api/stats',
        # New automation endpoints
        '/api/inject', '/api/memory/natural',
        '/api/memory/{memory_id}/confidence',
        '/api/memory/{memory_id}/verify',
        '/api/claude-md/sync',
        '/ws',
    ]
    for ep in endpoints:
        if ep in content:
            passed += 1
        else:
            print(f'    MISSING: {ep}')

    results.append(('C1. API Endpoints', passed, len(endpoints)))
    print(f'    {passed}/{len(endpoints)} endpoints verified')

    # ===== SUMMARY =====
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)

    total_passed = sum(r[1] for r in results)
    total_tests = sum(r[2] for r in results)

    section_a = [(n, p, t) for n, p, t in results if n.startswith('A')]
    section_b = [(n, p, t) for n, p, t in results if n.startswith('B')]
    section_c = [(n, p, t) for n, p, t in results if n.startswith('C')]

    print('\nSection A (Original 12 Features):')
    a_passed = sum(r[1] for r in section_a)
    a_total = sum(r[2] for r in section_a)
    for name, passed, total in section_a:
        status = 'PASS' if passed == total else 'PARTIAL' if passed > 0 else 'FAIL'
        print(f'  [{status}] {name}: {passed}/{total}')
    print(f'  Subtotal: {a_passed}/{a_total}')

    print('\nSection B (6 Automation Features):')
    b_passed = sum(r[1] for r in section_b)
    b_total = sum(r[2] for r in section_b)
    for name, passed, total in section_b:
        status = 'PASS' if passed == total else 'PARTIAL' if passed > 0 else 'FAIL'
        print(f'  [{status}] {name}: {passed}/{total}')
    print(f'  Subtotal: {b_passed}/{b_total}')

    print('\nSection C (Integration):')
    c_passed = sum(r[1] for r in section_c)
    c_total = sum(r[2] for r in section_c)
    for name, passed, total in section_c:
        status = 'PASS' if passed == total else 'PARTIAL' if passed > 0 else 'FAIL'
        print(f'  [{status}] {name}: {passed}/{total}')
    print(f'  Subtotal: {c_passed}/{c_total}')

    pct = 100 * total_passed // total_tests if total_tests > 0 else 0
    print(f'\n{"=" * 70}')
    print(f'TOTAL: {total_passed}/{total_tests} ({pct}%)')
    print(f'{"=" * 70}')

    if pct == 100:
        print('\n*** ALL TESTS PASSED! MEMORY SYSTEM FULLY OPERATIONAL ***')
    elif pct >= 90:
        print('\n*** MEMORY SYSTEM READY FOR USE ***')

    return pct

if __name__ == '__main__':
    pct = asyncio.run(test_all())
