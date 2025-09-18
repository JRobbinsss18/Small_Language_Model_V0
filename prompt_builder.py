from typing import List, Dict
from tokenizers import Tokenizer

class PromptBuilder:
    def build(self, tok: Tokenizer, docs: List[Dict[str,str]], question: str, max_ctx_tokens: int, spec_ids: dict) -> list[int]:
        header = "<|bos|><|system|>Answer using the CONTEXT only in 3–6 short bullet points. If not present, say you don't know."
        ctx = []
        for i, d in enumerate(docs, 1):
            ctx.append(f"[DOC{i} — {d['source']}]\n{d['text']}")
        context = "\n\n".join(ctx)
        prompt = f"{header}<|user|>CONTEXT:\n{context}\n\nQuestion: {question}<|assistant|>"
        ids = tok.encode(prompt).ids
        if len(ids) > max_ctx_tokens:
            ids = ids[-max_ctx_tokens:]
        return ids