

"""
Project 1 (Advanced AI): ChatGPT Clone
A web-based chat app with conversation memory, powered by a real LLM API
(Anthropic's Claude here; swapping to OpenAI is a few lines). This is
production-correct — add your API key and it works live.

KEY DESIGN POINTS (the things that make it a real chat app, not a toy):
  - Full conversation history is sent on every request (LLMs are stateless;
    "memory" = you resending the transcript each turn).
  - History is kept per browser session.
  - A MOCK mode (default when no API key is set) returns canned replies so
    the UI and message-flow are verifiable without a key or network.

Setup for real use:
    pip install flask anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python 01_chatgpt_clone.py

Concepts practiced: stateless-LLM conversation management, session-scoped
history, streaming-free request/response chat loop, graceful mock fallback.
"""

import os
from flask import Flask, request, jsonify, session, render_template_string

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"
MAX_HISTORY = 20  # cap turns sent, to control token cost


def call_llm(messages):
    """
    Real API call. `messages` is a list of {role, content} dicts.
    Falls back to a mock reply if no API key is configured.
    """
    if not API_KEY:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[MOCK REPLY — set ANTHROPIC_API_KEY for real responses] You said: '{last_user}'"

    # Real call (requires: pip install anthropic)
    import anthropic
    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=messages,
    )
    return response.content[0].text


PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Chat</title>
<style>
  body{margin:0;font-family:'Segoe UI',sans-serif;background:#0d0d12;color:#e8e8ee;height:100vh;display:flex;flex-direction:column}
  .header{padding:16px 24px;border-bottom:1px solid #222;font-weight:600}
  .messages{flex:1;overflow-y:auto;padding:24px;max-width:760px;margin:0 auto;width:100%;box-sizing:border-box}
  .msg{margin-bottom:18px;display:flex;gap:12px}
  .msg .who{font-size:.75rem;color:#888;width:60px;flex-shrink:0;padding-top:4px}
  .msg .bubble{background:#1a1b22;padding:12px 16px;border-radius:10px;line-height:1.5;white-space:pre-wrap}
  .msg.user .bubble{background:#2a3550}
  .input-row{display:flex;gap:10px;padding:16px 24px;border-top:1px solid #222;max-width:760px;margin:0 auto;width:100%;box-sizing:border-box}
  .input-row input{flex:1;padding:12px 16px;border-radius:10px;border:1px solid #2a2b33;background:#181920;color:#fff;outline:none}
  .input-row button{background:#6c8cff;color:#fff;border:none;padding:12px 22px;border-radius:10px;cursor:pointer;font-weight:600}
  .reset{float:right;color:#888;font-size:.8rem;cursor:pointer;background:none;border:none}
</style></head><body>
<div class="header">💬 Chat Clone <button class="reset" onclick="resetChat()">Clear</button></div>
<div class="messages" id="messages"></div>
<div class="input-row">
  <input id="input" placeholder="Type a message..." onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()">Send</button>
</div>
<script>
let history = [];
function render(){
  const box=document.getElementById('messages');box.innerHTML='';
  history.forEach(m=>{
    const d=document.createElement('div');d.className='msg '+m.role;
    d.innerHTML=`<div class="who">${m.role==='user'?'You':'AI'}</div><div class="bubble">${escapeHtml(m.content)}</div>`;
    box.appendChild(d);
  });
  box.scrollTop=box.scrollHeight;
}
async function send(){
  const inp=document.getElementById('input');const text=inp.value.trim();if(!text)return;
  history.push({role:'user',content:text});render();inp.value='';
  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
  const data=await res.json();
  history.push({role:'assistant',content:data.reply});render();
}
async function resetChat(){await fetch('/reset',{method:'POST'});history=[];render();}
function escapeHtml(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
</script></body></html>
"""


@app.route("/")
def index():
    session.setdefault("history", [])
    return render_template_string(PAGE)


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.get_json().get("message", "").strip()
    if not user_message:
        return jsonify({"reply": ""})

    history = session.get("history", [])
    history.append({"role": "user", "content": user_message})

    # Only send the most recent turns to control token cost.
    reply = call_llm(history[-MAX_HISTORY:])

    history.append({"role": "assistant", "content": reply})
    session["history"] = history
    session.modified = True
    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    print(f"LLM mode: {'REAL (API key set)' if API_KEY else 'MOCK (no key — set ANTHROPIC_API_KEY)'}")
    app.run(debug=True)