import os, re, torch
from tokenizers import Tokenizer
from slm_model import SmallLanguageModel
from tfidf_retriever import TfidfRetriever
from prompt_builder import PromptBuilder
from answerability_gate import AnswerabilityGate
from compressor import Compressor
from pdf_loader import PDFLoader

class RAGPipeline:
    def __init__(self):
        self.art='./artifacts'
        self.model_path=os.path.join(self.art,'slm.pt')
        self.retr_path=os.path.join(self.art,'retriever.pkl')
        self.doc_dir='./Documents'

    def _load_model_tok(self):
        ckpt=torch.load(self.model_path,map_location='cpu')
        cfg=ckpt['cfg']
        model=SmallLanguageModel(vocab_size=cfg['vocab_size'], context_length=cfg['context_length'], pad_id=cfg['pad_id'])
        model.load_state_dict(ckpt['state_dict'])
        device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device).eval()
        tok=Tokenizer.from_file(ckpt['tokenizer_path'])
        spec=ckpt['special_ids']
        return model,tok,spec,device

    def _token_chunks(self, tok, text: str, max_tokens=480, overlap=80):
        ids = tok.encode(text).ids
        out, i = [], 0
        while i < len(ids):
            piece = ids[i:i+max_tokens]
            out.append(tok.decode(piece))
            i += max(1, max_tokens - overlap)
        return out

    def _rebuild_retriever_with_sources(self, tok):
        items = PDFLoader().load(self.doc_dir)  # [{'text','source'}]
        chunks=[]
        for it in items:
            for ch in self._token_chunks(tok, it['text']):
                chunks.append({'text': ch, 'source': it['source']})
        retr = TfidfRetriever(self.retr_path)
        retr.rebuild(chunks)

    def answer(self, question: str, k: int = 5) -> str:
        model,tok,spec,device=self._load_model_tok()
        retr=TfidfRetriever(self.retr_path)
        retr.build_or_load(None)

        need_sources_fix = retr.legacy_upgraded or any(d.get('source')=='legacy' for d in (retr.docs or []))
        if need_sources_fix:
            self._rebuild_retriever_with_sources(tok)
            retr=TfidfRetriever(self.retr_path)
            retr.build_or_load(None)

        docs=retr.topk(question,k)
        if not AnswerabilityGate().is_answerable(question, docs):
            return "I don't know based on the provided PDFs."

        bullets = Compressor().extract(question, docs, max_points=5)
        if bullets:
            ans = "\n".join([f"- {b['text']} ({b['source']})" for b in bullets])
            return ans

        ids=PromptBuilder().build(tok, docs, question, max_ctx_tokens=800, spec_ids=spec)
        idx=torch.tensor([ids], dtype=torch.long, device=device)
        out=model.generate(idx, max_new_tokens=120, temperature=0.5, top_p=0.9, stop_ids=(spec['<|eos|>'],))
        text=tok.decode(out[0].tolist())
        m=re.search(r"<\\|assistant\\|>(.*)", text, re.DOTALL)
        ans=m.group(1).strip() if m else text
        ans=re.split(r"<\\|eos\\|>|<\\|user\\|>", ans)[0].strip()
        if not ans:
            return "I don't know based on the provided PDFs."
        lines=[l for l in ans.splitlines() if l.strip()]
        if len(lines)>6: lines=lines[:6]
        tail_sources = "; ".join(sorted({d['source'] for d in docs}))
        return "\n".join(lines) + f"\n\nSources: {tail_sources}"
