"""
Project 6 (Advanced AI): AI Teaching Assistant
Explains a topic at a chosen difficulty level, generates a quick quiz, and
tracks which concepts the learner struggled with — a personalized tutor.

Production-correct — add an API key for real explanations/quizzes. MOCK mode
(default) returns structured sample content so the flow is verifiable offline.

The interesting design element is ADAPTIVE DIFFICULTY: the same topic is
explained differently for "beginner" vs "advanced", driven by a level
parameter injected into the prompt — a simple, effective personalization
pattern.

Setup for real use:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python 06_ai_teaching_assistant.py

Concepts practiced: prompt parameterization for difficulty, structured
quiz generation (JSON), simple learner-progress tracking.
"""

import os
import json

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"


def call_llm(prompt, system=""):
    if not API_KEY:
        return None  # signal to caller to use mock content
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    resp = client.messages.create(model=MODEL, max_tokens=1024, system=system,
                                  messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


def explain_topic(topic, level):
    system = f"You are a patient tutor. Explain concepts at a {level} level."
    prompt = f"Explain '{topic}' in 3-4 sentences for a {level} learner."
    result = call_llm(prompt, system)
    if result is None:
        # Mock content that actually varies by level, to show the adaptation.
        samples = {
            "beginner": f"[MOCK] Think of {topic} like a simple everyday idea: it's a way to organize things step by step. Don't worry about the math yet — focus on the intuition first.",
            "advanced": f"[MOCK] {topic} can be formalized rigorously; consider its edge cases, complexity, and how it composes with related concepts in practice.",
        }
        return samples.get(level, samples["beginner"])
    return result


def generate_quiz(topic):
    prompt = (
        f"Create 2 multiple-choice questions about '{topic}'. Respond ONLY with "
        'JSON: [{"q":"...","options":["A","B","C","D"],"correct":0}]'
    )
    result = call_llm(prompt)
    if result is None:
        return [
            {"q": f"What is the main purpose of {topic}?",
             "options": ["To organize logic", "To slow programs down", "Nothing", "To use more memory"],
             "correct": 0},
            {"q": f"When would you use {topic}?",
             "options": ["Never", "When solving a structured problem", "Only in games", "Randomly"],
             "correct": 1},
        ]
    try:
        return json.loads(result.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return []


def main():
    print("=== AI Teaching Assistant ===")
    print(f"Mode: {'REAL' if API_KEY else 'MOCK (set ANTHROPIC_API_KEY for real content)'}\n")

    topic = input("What topic do you want to learn? ").strip() or "recursion"
    level = input("Level (beginner/advanced) [beginner]: ").strip().lower() or "beginner"

    print(f"\n====== Explain ({level}) ======")
    print(explain_topic(topic, level))

    print(f"\n--- Quick Quiz ---")
    quiz = generate_quiz(topic)
    struggled = []
    for i, question in enumerate(quiz, start=1):
        print(f"\nQ{i}. {question['q']}")
        for j, opt in enumerate(question["options"]):
            print(f"  {chr(65+j)}. {opt}")
        ans = input("Your answer (A/B/C/D): ").strip().upper()
        chosen = ord(ans) - 65 if ans in "ABCD" else -1
        if chosen == question["correct"]:
            print("  ✅ Correct!")
        else:
            correct_letter = chr(65 + question["correct"])
            print(f"  ❌ The answer was {correct_letter}.")
            struggled.append(question["q"])

    print("\n--- Progress ---")
    if struggled:
        print(f"Concepts to review ({len(struggled)}):")
        for q in struggled:
            print(f"  - {q}")
    else:
        print("Great job — you got everything right! 🎉")


if __name__ == "__main__":
    main()