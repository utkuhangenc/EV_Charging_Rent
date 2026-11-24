#!/usr/bin/env python3
"""Static pre-publication scan; prints only filenames/line numbers, never secrets."""
from pathlib import Path
import ast
import re

HERE = Path(__file__).resolve().parent
path_rules = [r"/depot/", r"/mnt/data/", r"/home/oai/"]
# Heuristic only: does not prove a file or Git history contains no secret.
key_rules = [
    r"(?i)(?:api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{12,}['\"]",
    r"(?i)(?:api[_-]?key|secret|token|password)\s*:\s*['\"][^'\"]{12,}['\"]",
]
errors = []
for file in sorted((HERE / "code").glob("*.py")):
    contents = file.read_text(encoding="utf-8")
    try:
        ast.parse(contents, filename=str(file))
    except SyntaxError as exc:
        errors.append((file.name, exc.lineno, "Syntax error"))
    for num, line in enumerate(contents.splitlines(), start=1):
        if any(re.search(p, line) for p in path_rules):
            errors.append((file.name, num, "Machine-specific path"))
        if any(re.search(p, line) for p in key_rules):
            errors.append((file.name, num, "Possible hardcoded credential"))
if errors:
    for file, line, issue in errors:
        print(f"REVIEW: {file}:{line} - {issue}")
    raise SystemExit(1)
print("PASS: Python syntax, fixed-machine-path and basic credential checks")
print("NOTE: Static checks cannot establish full reproducibility or detect all secrets.")
