import sys, py_compile
files = [
 r'src/retrieval_pipeline.py',
 r'src/stage2_rescorer.py',
 r'src/stage3_reranker.py',
 r'src/mcp_retrieval_server.py',
 r'run_mcp_server.py',
 r'run_benchmark.py',
]
ok = True
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print('OK:', f)
    except Exception as e:
        ok = False
        print('ERROR:', f, e)
sys.exit(0 if ok else 1)
