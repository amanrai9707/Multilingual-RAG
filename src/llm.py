"""LLM access. STATUS: the Groq wrapper is written from the Groq SDK docs and has NOT been run (needs your key).
Set the key as an environment variable - never in code or GitHub:   export GROQ_API_KEY=...   (in Colab: use Secrets)
Model names change; copy the current one from the Groq console and pass it with --llm-model."""
from __future__ import annotations
import os, re, time


def make_llm(model, temperature=0.0, max_tokens=512, retries=3):
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def llm(prompt):
        for attempt in range(retries):
            try:
                r = client.chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens,
                                                   messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content.strip()
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    return llm


def dry_llm(prompt):
    """Stand-in for --dry-run: echoes the first sentence of the first context passage. Proves the plumbing only."""
    if "Context:" not in prompt:
        return "I do not know."
    ctx = prompt.split("Context:", 1)[1].split("Question:", 1)[0].strip()
    if not ctx:
        return "INSUFFICIENT_INFORMATION"
    body = re.sub(r"^\[[^\]]*\]\s*", "", ctx.split("\n\n")[0])
    return re.split(r"(?<=[.!?])\s", body)[0][:240]
