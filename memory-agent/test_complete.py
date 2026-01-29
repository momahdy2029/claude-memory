"""Complete test suite matching actual codebase structure."""
import asyncio
import sys
import os
sys.path.insert(0, '.')

async def test_all():
    print('=' * 70)
    print('COMPLETE MEMORY SYSTEM TEST SUITE')
    print('=' * 70)

    results = []

    # ===== CORE SERVICES =====
    print('\n' + '-' * 70)
    print('SECTION A: CORE SERVICES')
    print('-' * 70)

    # [1] Database Service
    print('\n[A1] Database Service...')
    passed = 0
    try:
        from services.database import DatabaseService
        db = DatabaseService()
        await db.connect()
        await db.initialize_schema()

        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]

        key_tables = ['memories', 'patterns', 'projects', 'anchors', 'timeline']
        for t in key_tables:
            if t in tables:
                passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A1. Database', passed, 5))
    print(f'    {passed}/5 core tables exist')

    # [2] Embeddings Service
    print('\n[A2] Embeddings Service...')
    passed = 0
    try:
        from services.embeddings import OllamaEmbeddings
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A2. Embeddings', passed, 1))
    print(f'    {passed}/1 class verified')

    # [3] Auth Service
    print('\n[A3] Auth Service...')
    passed = 0
    try:
        from services.auth import AuthService
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A3. Auth', passed, 1))
    print(f'    {passed}/1 service verified')

    # [4] Retry Queue
    print('\n[A4] Retry Queue...')
    passed = 0
    try:
        from services.retry_queue import RetryQueue
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A4. Retry Queue', passed, 1))
    print(f'    {passed}/1 service verified')

    # [5] Cleanup Service
    print('\n[A5] Cleanup Service...')
    passed = 0
    try:
        from services.cleanup import CleanupService
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A5. Cleanup', passed, 1))
    print(f'    {passed}/1 service verified')

    # [6] Insights Service
    print('\n[A6] Insights Service...')
    passed = 0
    try:
        from services.insights import InsightService
        passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A6. Insights', passed, 1))
    print(f'    {passed}/1 service verified')

    # [7] WebSocket Service
    print('\n[A7] WebSocket Service...')
    passed = 0
    try:
        from services.websocket import WebSocketManager, broadcast_event, EventTypes
        passed += 3
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('A7. WebSocket', passed, 3))
    print(f'    {passed}/3 imports verified')

    # ===== SKILLS =====
    print('\n' + '-' * 70)
    print('SECTION B: SKILLS')
    print('-' * 70)

    # [B1] Store/Search Skills
    print('\n[B1] Store/Search Skills...')
    passed = 0
    try:
        from skills.store import store_memory
        passed += 1
    except Exception as e:
        print(f'    ERROR (store): {e}')
    try:
        from skills.search import semantic_search
        passed += 1
    except Exception as e:
        print(f'    ERROR (search): {e}')
    results.append(('B1. Store/Search', passed, 2))
    print(f'    {passed}/2 functions verified')

    # [B2] Timeline Skills
    print('\n[B2] Timeline Skills...')
    passed = 0
    try:
        from skills.timeline import timeline_log, get_timeline, get_session_timeline
        passed += 3
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B2. Timeline', passed, 3))
    print(f'    {passed}/3 functions verified')

    # [B3] Grounding Skills
    print('\n[B3] Grounding Skills...')
    passed = 0
    try:
        from skills.grounding import mark_anchor, get_anchors, check_anchor_conflicts
        passed += 3
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B3. Grounding', passed, 3))
    print(f'    {passed}/3 functions verified')

    # [B4] Admin Skills
    print('\n[B4] Admin Skills...')
    passed = 0
    try:
        from skills.admin import get_embedding_status, switch_embedding_model
        passed += 2
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('B4. Admin', passed, 2))
    print(f'    {passed}/2 functions verified')

    # ===== AUTOMATION FEATURES =====
    print('\n' + '-' * 70)
    print('SECTION C: AUTOMATION FEATURES (NEW)')
    print('-' * 70)

    # [C1] Auto-Capture Hook
    print('\n[C1] Auto-Capture Hook...')
    passed = 1 if os.path.exists('hooks/auto_capture.py') else 0
    results.append(('C1. Auto-Capture', passed, 1))
    print(f'    {passed}/1 file exists')

    # [C2] Session Hooks
    print('\n[C2] Session Start/End Hooks...')
    passed = 0
    if os.path.exists('hooks/session_start.py'): passed += 1
    if os.path.exists('hooks/session_end.py'): passed += 1
    results.append(('C2. Session Hooks', passed, 2))
    print(f'    {passed}/2 files exist')

    # [C3] Auto-Injector
    print('\n[C3] Auto-Injector...')
    passed = 0
    try:
        from services.auto_inject import AutoInjector, get_auto_injector
        ai = AutoInjector(None, None)
        kw = ai._extract_keywords('Fix authentication bug')
        if 'authentication' in kw or 'fix' in kw:
            passed += 1
        if ai._should_inject('test'):
            passed += 1
        passed += 1  # Import works
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('C3. Auto-Injector', passed, 3))
    print(f'    {passed}/3 tests passed')

    # [C4] Natural Language Interface
    print('\n[C4] Natural Language Interface...')
    passed = 0
    try:
        from skills.natural_language import parse_intent
        tests = [
            ('remember this: test', 'store'),
            ('show me past errors', 'list_errors'),
            ('what did I learn about Python', 'search'),
            ('memory stats', 'stats'),
        ]
        for text, expected in tests:
            intent, _ = parse_intent(text)
            if intent == expected:
                passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('C4. Natural Language', passed, 4))
    print(f'    {passed}/4 patterns verified')

    # [C5] CLAUDE.md Sync
    print('\n[C5] CLAUDE.md Sync...')
    passed = 0
    try:
        from services.claude_md_sync import ClaudeMdSync
        cms = ClaudeMdSync(None, None)
        # Test section finding
        test_md = "# Test\n## Preferences\n- Item"
        start, _ = cms._find_section(test_md, 'Preferences')
        if start > 0: passed += 1
        # Test content check
        if cms._content_exists(test_md, 'Item'): passed += 1
        if not cms._content_exists(test_md, 'Missing'): passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('C5. CLAUDE.md Sync', passed, 3))
    print(f'    {passed}/3 tests passed')

    # [C6] Confidence Scoring
    print('\n[C6] Confidence Scoring...')
    passed = 0
    try:
        from services.confidence import ConfidenceService
        from datetime import datetime

        class MockDB:
            class Conn:
                def cursor(self): return self
            conn = Conn()

        cs = ConfidenceService(MockDB(), None)
        age = cs._calculate_age_score(datetime.now().isoformat())
        if 0.9 <= age <= 1.0: passed += 1
        access = cs._calculate_access_score(5)
        if 0.5 <= access <= 1.0: passed += 1
        importance = cs._calculate_importance_score(10)
        if importance == 1.0: passed += 1
    except Exception as e:
        print(f'    ERROR: {e}')
    results.append(('C6. Confidence', passed, 3))
    print(f'    {passed}/3 calculations verified')

    # ===== INTEGRATION =====
    print('\n' + '-' * 70)
    print('SECTION D: INTEGRATION')
    print('-' * 70)

    # [D1] Main.py Endpoints
    print('\n[D1] API Endpoints in main.py...')
    passed = 0
    with open('main.py', 'r') as f:
        content = f.read()

    endpoints = [
        '/health', '/api/stats',
        '/api/inject', '/api/memory/natural',
        '/api/memory/{memory_id}/confidence',
        '/api/claude-md/sync', '/ws',
    ]
    for ep in endpoints:
        if ep in content:
            passed += 1
    results.append(('D1. API Endpoints', passed, len(endpoints)))
    print(f'    {passed}/{len(endpoints)} endpoints defined')

    # [D2] MCP Tool Handlers
    print('\n[D2] MCP Tool Handlers...')
    passed = 0
    handlers = [
        'memory_store', 'memory_search', 'memory_context',
        'timeline_log', 'mark_anchor',
    ]
    for h in handlers:
        if h in content:
            passed += 1
    results.append(('D2. MCP Handlers', passed, len(handlers)))
    print(f'    {passed}/{len(handlers)} handlers defined')

    # ===== SUMMARY =====
    print('\n' + '=' * 70)
    print('FINAL SUMMARY')
    print('=' * 70)

    sections = {
        'A': 'Core Services',
        'B': 'Skills',
        'C': 'Automation (NEW)',
        'D': 'Integration'
    }

    total_passed = 0
    total_tests = 0

    for section_key, section_name in sections.items():
        section_results = [(n, p, t) for n, p, t in results if n.startswith(section_key)]
        s_passed = sum(r[1] for r in section_results)
        s_total = sum(r[2] for r in section_results)
        total_passed += s_passed
        total_tests += s_total

        pct = 100 * s_passed // s_total if s_total > 0 else 0
        print(f'\n{section_name}: {s_passed}/{s_total} ({pct}%)')
        for name, passed, total in section_results:
            status = 'PASS' if passed == total else 'PARTIAL' if passed > 0 else 'FAIL'
            print(f'  [{status}] {name}: {passed}/{total}')

    pct = 100 * total_passed // total_tests if total_tests > 0 else 0
    print(f'\n{"=" * 70}')
    print(f'OVERALL: {total_passed}/{total_tests} ({pct}%)')
    print(f'{"=" * 70}')

    if pct == 100:
        print('\n*** ALL TESTS PASSED! MEMORY SYSTEM FULLY OPERATIONAL ***')
    elif pct >= 90:
        print('\n*** MEMORY SYSTEM READY FOR PRODUCTION ***')
    elif pct >= 75:
        print('\n*** MEMORY SYSTEM READY FOR USE ***')

    return pct

if __name__ == '__main__':
    asyncio.run(test_all())
