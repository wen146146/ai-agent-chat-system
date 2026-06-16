# -*- coding: utf-8 -*-
"""Phase 6 全量回归测试"""
import sys, os, io, json

sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

total = 0
passed = 0

def test(name, ok, detail=''):
    global total, passed
    total += 1
    if ok:
        passed += 1
        print(f'  [PASS] {name}')
    else:
        print(f'  [FAIL] {name} -- {detail}')

print('='*50)
print('Phase 6 - Regression Test Suite')
print('='*50)

# === Phase 1 ===
print('\n--- Phase 1: config_loader ---')
from src.utils.config_loader import resolve_safe_path, validate_write_path, validate_keys_at_startup
test('resolve_safe_path accepts "."', resolve_safe_path('.').startswith('C:'))
try:
    resolve_safe_path('../Windows')
    test('resolve_safe_path blocks traversal', False, 'should raise')
except PermissionError:
    test('resolve_safe_path blocks traversal', True)
warnings = validate_keys_at_startup()
test('validate_keys_at_startup runs', isinstance(warnings, list))

# === Phase 2 ===
print('\n--- Phase 2: New Tools ---')
from src.tools.file_tools import read_file, write_file, search_files, list_directory
from src.tools.shell_tools import run_command, _validate_command
from src.tools.app_tools import open_application, list_applications
from src.tools.system_tools import get_system_info, get_process_list
for name, fn in [('read_file', read_file), ('write_file', write_file), ('search_files', search_files),
                  ('list_directory', list_directory), ('run_command', run_command),
                  ('open_application', open_application), ('list_applications', list_applications),
                  ('get_system_info', get_system_info), ('get_process_list', get_process_list)]:
    test(f'{name} is callable', callable(fn))

# shell security
try:
    _validate_command('dir src/')
    test('shell: dir allowed', True)
except ValueError as e:
    test('shell: dir allowed', False, str(e))
try:
    _validate_command('del file.txt')
    test('shell: del blocked', False, 'should raise')
except ValueError:
    test('shell: del blocked', True)
try:
    _validate_command('python -c "import os"')
    test('shell: python -c blocked', False, 'should raise')
except ValueError:
    test('shell: python -c blocked', True)
try:
    _validate_command('pip install requests')
    test('shell: pip install blocked', False, 'should raise')
except ValueError:
    test('shell: pip install blocked', True)

# system_tools
result = get_system_info.invoke({})
test('get_system_info returns CPU info', 'CPU' in result)

# app_tools
result = list_applications.invoke({})
test('list_applications returns notepad', 'notepad' in result.lower())

# === Phase 3 ===
print('\n--- Phase 3: RAG Enhancement ---')
from src.knowledge_ingest.bm25_index import BM25Index
from src.knowledge_ingest.bm25_indexer import InvertedIndexBuilder
t1 = BM25Index._tokenize('Python闭包原理')
t2 = InvertedIndexBuilder.tokenize('Python闭包原理')
test('BM25 tokenize matches indexer', t1 == t2, f'{t1} vs {t2}')
bm25 = BM25Index()
test('BM25 not loaded before load', not bm25.is_loaded)

# RRF
from src.tools.rag_retriever_tool import RagRetrieveTool, RRF_K
parse = RagRetrieveTool.parse_rerank_response
test('RRF_K default=60', RRF_K == 60)
tool = RagRetrieveTool()
v = [{'content': 'A', 'chunk_id': 'a1', 'source': 'vector', 'score': 0.9}]
b = [{'content': 'B', 'chunk_id': 'b1', 'source': 'bm25', 'bm25_score': 12.5}]
k = [{'content': 'C', 'chunk_id': 'c1', 'source': 'keyword', 'score': 0.8}]
fused = tool.fuse_results(v, b, k, top_k=3)
test('RRF returns 3', len(fused) == 3)
test('RRF sorted', all(fused[i]['rrf_score'] >= fused[i+1]['rrf_score'] for i in range(len(fused)-1)))
v2 = [{'content': 'X', 'chunk_id': 'x1', 'source': 'vector', 'score': 0.9}]
b2 = [{'content': 'X', 'chunk_id': 'x1', 'source': 'bm25', 'bm25_score': 10.0}]
fused2 = tool.fuse_results(v2, b2, [], top_k=5)
test('RRF dedup', len(fused2) == 1)

# rerank JSON parsing
BT = chr(96)
text = BT*3 + 'json\n{"ranked_indices": [4, 2, 0]}\n' + BT*3
test('rerank: codeblock JSON', parse(text, 5) == [4, 2, 0])
test('rerank: bare array', parse('[3, 1, 2]', 5) == [2, 0, 1])
test('rerank: comma sep', parse('3, 1, 2', 5) == [2, 0, 1])
test('rerank: natural text', parse('文档3和文档1', 5) == [2, 0])
test('rerank: fallback', parse('无意义', 3) == [0, 1, 2])

# === Phase 4 ===
print('\n--- Phase 4: Agent Optimization ---')
import src.agent.agent_chain as ac
test('MAX_AGENT_ITERATIONS=12', ac.MAX_AGENT_ITERATIONS == 12)
test('SYSTEM_PROMPT has role', '面试知识助手' in ac.SYSTEM_PROMPT)
test('SYSTEM_PROMPT has source mention', '引用来源' in ac.SYSTEM_PROMPT)

from src.utils.audit import _sanitize_args
test('audit: password masked', _sanitize_args({'password': 'x'})['password'] == '***')
test('audit: token masked', _sanitize_args({'token': 'x'})['token'] == '***')
test('audit: API-KEY masked', _sanitize_args({'API-KEY': 'x'})['API-KEY'] == '***')
test('audit: ApiKey matched', _sanitize_args({'ApiKey': 'x'})['ApiKey'] == '***')
test('audit: content intact', _sanitize_args({'content': 'hello'})['content'] == 'hello')

# === Phase 5 ===
print('\n--- Phase 5: Frontend ---')
with open('static/chat.html', 'r', encoding='utf-8') as f:
    html = f.read()
test('source-card CSS', '.source-card' in html)
test('cursor-blink CSS', 'cursor-blink' in html)
test('restoreSession JS', 'restoreSession' in html)
test('createNewSession JS', 'createNewSession' in html)
test('chatKbFilter HTML', 'chatKbFilter' in html)
test('renderSourceCard JS', 'renderSourceCard' in html)
test('toggleSource JS', 'toggleSource' in html)
test('final_output SSE handling', 'final_output' in html)
test('</html> closing tag', '</html>' in html)
test('</script> closing tag', '</script>' in html)

# === ToolRegistry ===
print('\n--- ToolRegistry ---')
from src.tools.tool_registry import ToolRegistry
r = ToolRegistry.get_instance()
tools = r.collect_native_tools()
names = [t.name for t in tools]
test('total tools = 20', len(tools) == 20, f'got {len(tools)}')
required = ['calculator', 'web_search', 'web_fetch', 'rag_retrieve',
            'episodic_memory_save', 'episodic_memory_search', 'semantic_memory_save',
            'semantic_memory_search', 'semantic_memory_delete', 'semantic_memory_count',
            'read_file', 'write_file', 'search_files', 'list_directory',
            'run_command', 'open_application', 'list_applications',
            'get_system_info', 'get_process_list']
for name in required:
    test(f'registered: {name}', name in names)

# === Actual Tool Invocation Tests ===
print('\n--- Actual Tool Invocation ---')

# file_tools
from src.tools.file_tools import read_file, write_file, search_files, list_directory
r = read_file.invoke({"path": "README.md", "max_bytes": 30000})
test('read_file README.md', 'AI Agent' in r)
r = search_files.invoke({"pattern": "*.py", "root_dir": "./src/tools"})
test('search_files *.py', 'calculator_tool.py' in r)
r = list_directory.invoke({"path": "./src/tools"})
test('list_directory src/tools', 'calculator_tool.py' in r)
r = write_file.invoke({"path": "./data/_test_write.txt", "content": "test", "mode": "overwrite"})
test('write_file creates file', '成功' in r or '成功' in r)
import os
if os.path.exists('./data/_test_write.txt'):
    os.remove('./data/_test_write.txt')
    if os.path.isdir('./data') and not os.listdir('./data'):
        os.rmdir('./data')

# shell_tools
from src.tools.shell_tools import run_command
r = run_command.invoke({"command": "echo hello", "timeout": 5})
test('run_command echo', 'hello' in r)
r = run_command.invoke({"command": "dir /b", "cwd": "src/tools", "timeout": 5})
test('run_command dir /b', 'calculator_tool.py' in r)
r = run_command.invoke({"command": "git --version", "timeout": 5})
test('run_command git', 'git' in r.lower())
r = run_command.invoke({"command": "del test.txt", "timeout": 5})
test('run_command del blocked', '拦截' in r or 'blocked' in r.lower())
r = run_command.invoke({"command": "python -c \"import os\"", "timeout": 5})
test('run_command python -c blocked', '拦截' in r or 'blocked' in r.lower())

# system_tools
from src.tools.system_tools import get_system_info, get_process_list
r = get_system_info.invoke({})
test('get_system_info CPU', 'CPU' in r)
r = get_process_list.invoke({"top_n": 5})
test('get_process_list PID', 'PID' in r)

# app_tools
from src.tools.app_tools import list_applications
r = list_applications.invoke({})
test('list_applications notepad', 'notepad' in r.lower())

# === Summary ===
print(f'\n{"="*50}')
print(f'Results: {passed}/{total} passed')
if passed == total:
    print('ALL TESTS PASSED')
else:
    print(f'{total - passed} FAILURES')
print(f'{"="*50}')
