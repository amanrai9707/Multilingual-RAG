"""Assamese <-> English translation with IndicTrans2.

STATUS: written from the IndicTrans2 model card as I remember it; NOT executed (the build
environment cannot download Hugging Face models). Treat as a starting point and compare against
the CURRENT model card at https://huggingface.co/ai4bharat/indictrans2-en-indic-1B before relying
on it - install instructions for IndicTransToolkit in particular change over time.

Language codes IndicTrans2 uses: Assamese = "asm_Beng", English = "eng_Latn".
Two separate checkpoints exist: en->indic and indic->en.
"""
from __future__ import annotations

EN, AS = "eng_Latn", "asm_Beng"
CKPT = {"en-indic": "ai4bharat/indictrans2-en-indic-1B", "indic-en": "ai4bharat/indictrans2-indic-en-1B"}


class IndicTranslator:
    def __init__(self, direction: str, device: str | None = None):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor  # pip install IndicTransToolkit (see model card)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.direction = direction
        self.src, self.tgt = (EN, AS) if direction == "en-indic" else (AS, EN)
        name = CKPT[direction]
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).to(self.device)
        self.model.eval()
        self.ip = IndicProcessor(inference=True)

    def __call__(self, text: str, num_beams: int = 5, max_length: int = 256) -> str:
        batch = self.ip.preprocess_batch([text], src_lang=self.src, tgt_lang=self.tgt)
        inputs = self.tok(batch, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=max_length,
                                      num_beams=num_beams, num_return_sequences=1)
        decoded = self.tok.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return self.ip.postprocess_batch(decoded, lang=self.tgt)[0]


def make_translators(device: str | None = None):
    """Returns (assamese_to_english, english_to_assamese) callables."""
    return IndicTranslator("indic-en", device), IndicTranslator("en-indic", device)
