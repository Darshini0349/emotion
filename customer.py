"""
Project 5 (Advanced AI): AI Customer Support Bot
A support bot that answers questions grounded in a knowledge base (FAQ
entries), rather than making things up. This is a lightweight preview of
RETRIEVAL-AUGMENTED GENERATION (RAG): find the most relevant KB entries,
then have the LLM answer using ONLY those.

Why grounding matters: an ungrounded LLM will confidently invent a refund
policy. Grounding it in your real KB keeps answers accurate and lets the
bot say "I don't know, contact support" when the KB doesn't cover something.

Production-correct — add an API key for real answers. MOCK mode (default)
returns the retrieved KB text so you can see the retrieval working offline.

Setup for real use:
    pip install anthropic scikit-learn
    export ANTHROPIC_API_KEY=sk-ant-...
    python 05_ai_customer_support_bot.py

Concepts practiced: retrieval via TF-IDF cosine similarity, grounding an
LLM in retrieved context, refusing to answer outside the KB, the RAG
pattern in miniature (which the RAG track expands with vector DBs).
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"
SIMILARITY_THRESHOLD = 0.1  # below this, we treat the KB as not covering the question

KNOWLEDGE_BASE = [
    {"q": "What is your refund policy?",
     "a": "We offer full refunds within 30 days of purchase, no questions asked. After 30 days, refunds are handled case by case."},
    {"q": "How long does shipping take?",
     "a": "Standard shipping takes 5-7 business days. Express shipping (extra charge) takes 2-3 business days."},
    {"q": "Do you ship internationally?",
     "a": "Yes, we ship to most countries. International orders take 10-15 business days and may incur customs fees."},
    {"q": "How do I reset my password?",
     "a": "Click 'Forgot Password' on the login page and follow the emailed link. The link expires in 1 hour."},
    {"q": "How do I contact human support?",
     "a": "Email support@example.com or call 1-800-555-0100, Monday to Friday, 9am-6pm."},
    {"q": "Can I change my order after placing it?",
     "a": "You can modify or cancel an order within 1 hour of placing it, from your Orders page."},
]


# A tiny synonym map so common paraphrases still retrieve. In a REAL system
# you would NOT do this by hand — you'd use semantic embeddings (e.g.
# sentence-transformers) whose vectors already place "refund" and "money
# back" close together. This keyword+synonym approach is a transparent stand-in
# that shows WHY embeddings are worth it: keyword matching alone scores 0.0
# when the user's words differ from the KB's words.
SYNONYMS = {
    "money back": "refund", "money": "refund", "reimburse": "refund",
    "arrive": "shipping delivery", "package": "shipping order",
    "delivery": "shipping", "sign in": "password login",
    "cancel": "change order", "modify": "change order",
    "phone": "contact support", "call": "contact support", "human": "contact support",
}


def expand_query(text):
    """Expand paraphrases toward KB vocabulary (a manual stand-in for embeddings)."""
    expanded = text.lower()
    for phrase, replacement in SYNONYMS.items():
        if phrase in expanded:
            expanded += " " + replacement
    return expanded


class SupportBot:
    def __init__(self, kb):
        self.kb = kb
        # Index BOTH the question and answer text — more vocabulary to match against.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.kb_matrix = self.vectorizer.fit_transform(
            [entry["q"] + " " + entry["a"] for entry in kb]
        )

    def retrieve(self, query, top_k=2):
        query_vec = self.vectorizer.transform([expand_query(query)])
        sims = cosine_similarity(query_vec, self.kb_matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda x: -x[1])
        results = [(self.kb[i], score) for i, score in ranked[:top_k] if score > SIMILARITY_THRESHOLD]
        return results

    def answer(self, query):
        retrieved = self.retrieve(query)
        if not retrieved:
            return ("I'm sorry, I don't have information on that. Please contact "
                    "support@example.com and a human will help you.")

        context = "\n".join(f"Q: {entry['q']}\nA: {entry['a']}" for entry, _ in retrieved)
        return self._generate(query, context)

    def _generate(self, query, context):
        if not API_KEY:
            # Mock: return the top retrieved KB answer directly.
            top_entry = self.retrieve(query)[0][0]
            return f"[MOCK — grounded in KB] {top_entry['a']}"
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)
        prompt = (
            "You are a support agent. Answer the customer's question using ONLY "
            "the knowledge base below. If it doesn't contain the answer, say you "
            f"don't know and to contact support.\n\nKNOWLEDGE BASE:\n{context}\n\n"
            f"CUSTOMER: {query}"
        )
        resp = client.messages.create(model=MODEL, max_tokens=512,
                                      messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text


def main():
    print("=== AI Customer Support Bot ===")
    print(f"Mode: {'REAL LLM' if API_KEY else 'MOCK (retrieval is real)'}")
    bot = SupportBot(KNOWLEDGE_BASE)

    # Demo a few questions, including one the KB can't answer.
    test_questions = [
        "How do I get my money back?",
        "When will my package arrive?",
        "Do you sell gift cards?",  # not in KB -> should refuse
    ]

    for q in test_questions:
        print(f"\nCustomer: {q}")
        print(f"Bot: {bot.answer(q)}")


if __name__ == "__main__":
    main()