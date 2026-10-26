
"""
Project 2 (Advanced AI): AI Interview Assistant
A CLI tool that conducts a mock interview: it generates role-specific
questions, takes the candidate's answers, and gives structured feedback
(strengths, gaps, a score) using an LLM.

Production-correct — add an API key and it runs live. Ships with a MOCK
mode (default, no key) so the flow is verifiable offline.

The interesting part is PROMPT ENGINEERING: the feedback prompt asks for a
strict JSON schema so the reply is machine-parseable, not free text — a
key skill for building reliable LLM apps.

Setup for real use:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python 02_ai_interview_assistant.py

Concepts practiced: structured-output prompting (JSON schema), multi-turn
interview flow, parsing + validating LLM JSON responses, mock fallback.
"""

import os
import json

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

ROLE_QUESTIONS = {
    "backend": [
        "Explain the difference between SQL and NoSQL databases, and when you'd choose each.",
        "How would you design a rate limiter for an API?",
        "What happens, step by step, when a database transaction is committed?",
    ],
    "ml": [
        "How do you handle class imbalance in a classification problem?",
        "Explain the bias-variance tradeoff in your own words.",
        "Walk me through how you'd deploy a trained model to production.",
    ],
}


def call_llm(prompt, system=""):
    if not API_KEY:
        # Mock: return a canned structured evaluation so the flow works offline.
        return json.dumps({
            "score": 7,
            "strengths": ["Clear explanation", "Used a concrete example"],
            "gaps": ["Could mention trade-offs more explicitly"],
            "follow_up": "Can you give a specific example from a past project?",
        })
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def evaluate_answer(question, answer):
    system = (
        "You are a strict but fair technical interviewer. Evaluate the "
        "candidate's answer. Respond with ONLY valid JSON, no prose, in this "
        'schema: {"score": <1-10>, "strengths": [..], "gaps": [..], '
        '"follow_up": "<one follow-up question>"}'
    )
    prompt = f"Question: {question}\n\nCandidate's answer: {answer}\n\nEvaluate."
    raw = call_llm(prompt, system=system)
    try:
        # Strip accidental markdown fences before parsing.
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"score": None, "strengths": [], "gaps": ["Could not parse model output"], "follow_up": raw}


def run_interview(role):
    questions = ROLE_QUESTIONS.get(role, ROLE_QUESTIONS["backend"])
    total_score = 0
    scored = 0

    print(f"\n=== Mock Interview: {role.upper()} role ===")
    print(f"{len(questions)} questions. Type your answer after each.\n")

    for i, question in enumerate(questions, start=1):
        print(f"Q{i}. {question}")
        answer = input("Your answer: ").strip()
        if not answer:
            answer = "(no answer)"

        feedback = evaluate_answer(question, answer)
        print(f"\n  Score: {feedback['score']}/10")
        if feedback["strengths"]:
            print(f"  Strengths: {', '.join(feedback['strengths'])}")
        if feedback["gaps"]:
            print(f"  To improve: {', '.join(feedback['gaps'])}")
        if feedback.get("follow_up"):
            print(f"  Follow-up: {feedback['follow_up']}")
        print()

        if isinstance(feedback["score"], (int, float)):
            total_score += feedback["score"]
            scored += 1

    if scored:
        print(f"=== Interview complete. Average score: {total_score/scored:.1f}/10 ===")


def main():
    print("=== AI Interview Assistant ===")
    print(f"Mode: {'REAL' if API_KEY else 'MOCK (set ANTHROPIC_API_KEY for real evaluation)'}")
    print("Available roles: backend, ml")
    role = input("Choose a role: ").strip().lower() or "backend"
    run_interview(role)


if __name__ == "__main__":
    main()
