
"""
Project 4 (Advanced AI): AI Code Reviewer
Reviews a Python file and produces feedback: it combines fast, deterministic
STATIC CHECKS (pure Python — no LLM) with an LLM pass for deeper review.

Same cost-aware design as the resume analyzer: catch the cheap, certain
issues with rules; use the LLM only for the judgement calls it's actually
needed for.

Production-correct — add an API key for the LLM pass. MOCK mode (default)
runs the static checks (which are 100% real) and returns a canned LLM review.

Setup for real use:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python 04_ai_code_reviewer.py path/to/file.py

Concepts practiced: AST-free static analysis with regex/heuristics, combining
rule-based + LLM review, presenting findings by severity.
"""

import os
import sys
import re

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"


def static_checks(code):
    """Real, deterministic checks — no LLM needed."""
    findings = []
    lines = code.split("\n")

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Bare except is an anti-pattern.
        if re.match(r"except\s*:", stripped):
            findings.append((i, "warning", "Bare 'except:' catches everything — specify an exception type."))

        # Mutable default arguments are a classic Python bug.
        if re.search(r"def \w+\([^)]*=\s*(\[\]|\{\})", line):
            findings.append((i, "error", "Mutable default argument ([] or {}) — use None and create inside."))

        # eval/exec are dangerous.
        if re.search(r"\b(eval|exec)\s*\(", stripped):
            findings.append((i, "error", f"Use of {stripped.split('(')[0].split()[-1]}() is a security risk."))

        # Long lines.
        if len(line) > 100:
            findings.append((i, "info", f"Line is {len(line)} chars (>100) — consider wrapping."))

        # TODO/FIXME markers.
        if re.search(r"#\s*(TODO|FIXME)", line, re.IGNORECASE):
            findings.append((i, "info", "Unresolved TODO/FIXME marker."))

        # print() left in (often debug leftovers).
        if re.match(r"print\s*\(", stripped):
            findings.append((i, "info", "print() call — is this debug output that should be removed/logged?"))

    return findings


def call_llm(code):
    if not API_KEY:
        return ("[MOCK LLM REVIEW] Overall structure is reasonable. Consider adding "
                "docstrings to public functions, type hints for clarity, and unit "
                "tests for the core logic. Set ANTHROPIC_API_KEY for a real review.")
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    prompt = (
        "Review this Python code as a senior engineer. Give concise, actionable "
        f"feedback on design, readability, and bugs:\n\n```python\n{code}\n```"
    )
    resp = client.messages.create(model=MODEL, max_tokens=1024,
                                  messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


SAMPLE_CODE = '''
def add_item(item, basket=[]):      # mutable default bug
    basket.append(item)
    return basket

def risky(user_input):
    result = eval(user_input)        # security risk
    try:
        return result
    except:                          # bare except
        pass

print("debugging here")             # leftover debug print
'''


def main():
    print("=== AI Code Reviewer ===")
    print(f"Mode: {'REAL LLM' if API_KEY else 'MOCK LLM (static checks are real)'}\n")

    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            code = f.read()
        print(f"Reviewing: {sys.argv[1]}")
    else:
        code = SAMPLE_CODE
        print("No file given — reviewing built-in sample with known issues.")

    findings = static_checks(code)
    print(f"\n--- Static analysis: {len(findings)} finding(s) ---")
    severity_order = {"error": 0, "warning": 1, "info": 2}
    for line_no, severity, message in sorted(findings, key=lambda f: (severity_order[f[1]], f[0])):
        print(f"  [{severity.upper():7}] line {line_no}: {message}")

    print("\n--- LLM review ---")
    print(f"  {call_llm(code)}")


if __name__ == "__main__":
    main()