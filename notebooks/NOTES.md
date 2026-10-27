# Colab notes
- Groq call (check current model names in the Groq console; they change):
    from groq import Groq
    client = Groq(api_key="YOUR_KEY")      # never commit the key to GitHub
    def llm(prompt):
        r = client.chat.completions.create(model="<current model name>", messages=[{"role":"user","content":prompt}], temperature=0)
        return r.choices[0].message.content
  Use temperature=0 so repeated runs give the same answer (needed for a fair comparison).
- Fix k (number of retrieved chunks) once, e.g. k=5, before the final run. Do not change per system.
- Save results/raw_log.jsonl to Drive after every run.
