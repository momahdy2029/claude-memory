"""Test suite for automation features."""
import asyncio
import sys
sys.path.insert(0, '.')

async def test_all():
    print('=' * 60)
    print('AUTOMATION FEATURES TEST SUITE')
    print('=' * 60)

    results = []

    # Test 1: Natural Language Parser
    print('\n[1] Testing Natural Language Parser...')
    from skills.natural_language import parse_intent
    tests = [
        ('remember this: testing works', 'store'),
        ('what did I learn about Python', 'search'),
        ('show me past errors', 'list_errors'),
        ('show me decisions', 'list_decisions'),
        ('memory stats', 'stats'),
        ('forget about old stuff', 'forget'),
        ('show me patterns', 'list_patterns'),
        ('project info', 'project_context'),
        ('search for authentication', 'search'),
    ]

    passed = 0
    for text, expected_intent in tests:
        intent, _ = parse_intent(text)
        if intent == expected_intent:
            passed += 1
        else:
            print(f'    FAIL: "{text}" -> got {intent}, expected {expected_intent}')

    results.append(('Natural Language Parser', passed, len(tests)))
    print(f'    {passed}/{len(tests)} patterns correct')

    # Test 2: Confidence Service
    print('\n[2] Testing Confidence Service...')
    from services.confidence import ConfidenceService
    from datetime import datetime

    class MockDB:
        class Conn:
            def cursor(self): return self
            def execute(self, *args): pass
            def fetchone(self): return None
            def fetchall(self): return []
            def commit(self): pass
        conn = Conn()

    cs = ConfidenceService(MockDB(), None)

    age_score = cs._calculate_age_score(datetime.now().isoformat())
    access_score = cs._calculate_access_score(10)
    importance_score = cs._calculate_importance_score(8)

    passed = 0
    if 0.9 <= age_score <= 1.0: passed += 1
    else: print(f'    FAIL: age_score = {age_score}, expected 0.9-1.0')
    if 0.5 <= access_score <= 1.0: passed += 1
    else: print(f'    FAIL: access_score = {access_score}, expected 0.5-1.0')
    if importance_score == 0.8: passed += 1
    else: print(f'    FAIL: importance_score = {importance_score}, expected 0.8')

    results.append(('Confidence Scoring', passed, 3))
    print(f'    {passed}/3 calculations correct')

    # Test 3: Auto Injector
    print('\n[3] Testing Auto Injector...')
    from services.auto_inject import AutoInjector

    ai = AutoInjector(MockDB(), None)

    # Test keyword extraction
    keywords = ai._extract_keywords('Fix the authentication bug in login.py')
    passed = 0
    if 'authentication' in keywords or 'login' in keywords: passed += 1
    else: print(f'    FAIL: Expected auth keywords, got {keywords}')

    # Test should_inject logic
    should = ai._should_inject('test query')
    if should: passed += 1
    else: print(f'    FAIL: Should inject on first query')

    # Test format
    test_context = {
        'injected': True,
        'patterns': [{'name': 'Test Pattern', 'solution': 'Test solution'}],
        'memories': [{'type': 'decision', 'content': 'Decision 1'}],
        'warnings': []
    }
    formatted = ai.format_injection(test_context)
    if 'Test Pattern' in formatted: passed += 1
    else: print(f'    FAIL: Pattern not in formatted output')

    results.append(('Auto Injector', passed, 3))
    print(f'    {passed}/3 tests correct')

    # Test 4: CLAUDE.md Sync
    print('\n[4] Testing CLAUDE.md Sync...')
    from services.claude_md_sync import ClaudeMdSync

    cms = ClaudeMdSync(MockDB(), None)

    test_md = """# Header
## Preferences
- Pref 1
- Pref 2

## Other Section
Content here
"""
    start, end = cms._find_section(test_md, 'Preferences')
    passed = 0
    if start > 0: passed += 1
    else: print(f'    FAIL: Could not find Preferences section')

    if cms._content_exists(test_md, 'Pref 1'): passed += 1
    else: print(f'    FAIL: Should detect existing content')

    if not cms._content_exists(test_md, 'New Content'): passed += 1
    else: print(f'    FAIL: Should not detect non-existing content')

    results.append(('CLAUDE.md Sync', passed, 3))
    print(f'    {passed}/3 tests correct')

    # Test 5: Hook Files Exist
    print('\n[5] Testing Hook Files...')
    import os
    hook_dir = 'hooks'
    hooks = ['auto_capture.py', 'session_start.py', 'session_end.py']
    passed = 0
    for hook in hooks:
        path = os.path.join(hook_dir, hook)
        if os.path.exists(path):
            passed += 1
        else:
            print(f'    FAIL: {path} not found')

    results.append(('Hook Files', passed, len(hooks)))
    print(f'    {passed}/{len(hooks)} hooks exist')

    # Test 6: Service Imports
    print('\n[6] Testing Service Imports...')
    passed = 0
    try:
        from services.auto_inject import get_auto_injector
        passed += 1
    except Exception as e:
        print(f'    FAIL: auto_inject import: {e}')

    try:
        from services.confidence import get_confidence_service
        passed += 1
    except Exception as e:
        print(f'    FAIL: confidence import: {e}')

    try:
        from services.claude_md_sync import get_claude_md_sync
        passed += 1
    except Exception as e:
        print(f'    FAIL: claude_md_sync import: {e}')

    try:
        from skills.natural_language import process_natural_command
        passed += 1
    except Exception as e:
        print(f'    FAIL: natural_language import: {e}')

    results.append(('Service Imports', passed, 4))
    print(f'    {passed}/4 imports successful')

    # Test 7: Main.py Endpoint Integration
    print('\n[7] Testing Main.py Endpoints...')
    passed = 0

    with open('main.py', 'r') as f:
        main_content = f.read()

    endpoints_to_check = [
        '/api/inject',
        '/api/memory/natural',
        '/api/memory/{memory_id}/confidence',
        '/api/memory/{memory_id}/verify',
        '/api/claude-md/sync',
        '/api/claude-md/suggestions',
    ]

    for endpoint in endpoints_to_check:
        if endpoint in main_content:
            passed += 1
        else:
            print(f'    FAIL: Endpoint {endpoint} not found in main.py')

    results.append(('Main.py Endpoints', passed, len(endpoints_to_check)))
    print(f'    {passed}/{len(endpoints_to_check)} endpoints defined')

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    total_passed = sum(r[1] for r in results)
    total_tests = sum(r[2] for r in results)

    for name, passed, total in results:
        status = 'PASS' if passed == total else 'PARTIAL' if passed > 0 else 'FAIL'
        print(f'  [{status}] {name}: {passed}/{total}')

    pct = 100*total_passed//total_tests
    print(f'\nTotal: {total_passed}/{total_tests} ({pct}%)')

    if pct == 100:
        print('\n*** ALL AUTOMATION TESTS PASSED! ***')

    return total_passed == total_tests

if __name__ == '__main__':
    asyncio.run(test_all())
