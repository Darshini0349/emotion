"""
Project 3 (Advanced AI): AI Resume Analyzer
Analyzes how well a resume matches a job description: computes a match
score, lists matching + missing skills, and suggests concrete
improvements. Uses an LLM for the judgement, structured as JSON.

Production-correct — add an API key for real analysis. MOCK mode (default)
returns a realistic structured result so the flow is verifiable offline.

Combines two techniques:
  1. A cheap, deterministic KEYWORD overlap (pure Python) for a baseline
     skills match — no LLM needed, instant.
  2. An LLM pass for nuanced feedback the keyword match can't give.
This "cheap check + LLM for the hard part" split is good real-world design.

Setup for real use:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python 03_ai_resume_analyzer.py

Concepts practiced: hybrid rule-based + LLM analysis, keyword extraction,
structured JSON output, cost-aware design (don't use an LLM for what a set
intersection can do).
"""

import os
import re
import json

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

# A small skills vocabulary for the deterministic keyword pass.
KNOWN_SKILLS = {
    "python", "java", "c++", "javascript", "sql", "postgresql", "mysql",
    "flask", "django", "fastapi", "react", "node", "docker", "kubernetes",
    "aws", "git", "machine learning", "tensorflow", "pytorch", "pandas",
    "numpy", "scikit-learn", "nlp", "computer vision", "rest api", "html",
    "css", "linux", "mongodb", "redis", "graphql",
}


def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in KNOWN_SKILLS:
        # word-boundary-ish match so "java" doesn't match "javascript"
        if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
            found.add(skill)
    return found


def keyword_match(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)
    if not job_skills:
        return 0, set(), set()
    matched = resume_skills & job_skills
    missing = job_skills - resume_skills
    score = round(len(matched) / len(job_skills) * 100)
    return score, matched, missing


def call_llm(prompt):
    if not API_KEY:
        return json.dumps({
            "overall_assessment": "[MOCK] Solid technical foundation; tailor wording to the role.",
            "suggestions": [
                "Quantify impact (e.g. 'reduced load time by 40%')",
                "Move most-relevant projects to the top",
                "Add the missing skills if you have them, even from coursework",
            ],
        })
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def llm_feedback(resume_text, job_text, missing_skills):
    prompt = (
        "You are a resume coach. Given the resume and job description, respond "
        "with ONLY JSON: {\"overall_assessment\": \"...\", \"suggestions\": [\"...\"]}. "
        f"\n\nRESUME:\n{resume_text}\n\nJOB:\n{job_text}\n\n"
        f"Missing skills detected: {', '.join(missing_skills) or 'none'}."
    )
    raw = call_llm(prompt)
    try:
        return json.loads(raw.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return {"overall_assessment": raw, "suggestions": []}


SAMPLE_RESUME = """
Darshini N — Software Engineer
Skills: Python, C, PostgreSQL, Flask, FastAPI, HTML, CSS, JavaScript, Git
Projects: Built a crop-monitoring platform with FastAPI and satellite data.
Built LifeGuard AI, a Flask disease-prediction app. Trained ML classifiers
with scikit-learn and pandas.
"""

SAMPLE_JOB = """
We're hiring a Backend/ML Engineer. Required: Python, SQL, REST API design,
Flask or FastAPI, Docker, machine learning, pandas. Nice to have: AWS,
Kubernetes, React.
"""


def main():
    print("=== AI Resume Analyzer ===")
    print(f"Mode: {'REAL' if API_KEY else 'MOCK (set ANTHROPIC_API_KEY for real feedback)'}\n")

    resume, job = SAMPLE_RESUME, SAMPLE_JOB

    # Step 1: fast deterministic keyword match.
    score, matched, missing = keyword_match(resume, job)
    print(f"Skills match score: {score}%")
    print(f"  Matched skills: {', '.join(sorted(matched)) or 'none'}")
    print(f"  Missing skills: {', '.join(sorted(missing)) or 'none'}")

    # Step 2: LLM for nuanced advice.
    print("\nAI feedback:")
    feedback = llm_feedback(resume, job, missing)
    print(f"  {feedback['overall_assessment']}")
    for i, s in enumerate(feedback.get("suggestions", []), start=1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    main()