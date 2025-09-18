import os, math, numpy as np, torch
from torch.utils.data import DataLoader, random_split
from pdf_loader import PDFLoader
from bpe_tokenizer import BPETokenizer
from text_dataset import TextDataset
from slm_model import SmallLanguageModel
from tfidf_retriever import TfidfRetriever

class Trainer:
    def __init__(self):
        self.art = './artifacts'
        self.doc = './Documents'
        self.tokenizer_path = os.path.join(self.art, 'tokenizer.json')
        self.retriever_path = os.path.join(self.art, 'retriever.pkl')
        self.model_path = os.path.join(self.art, 'slm.pt')

    def make_chat_corpus(self, docs: list[dict]) -> str:
        sys = "Answer questions using the CONTEXT. If not in context, say you don't know."
        ex = []
        for d in docs:
            ctx = d['text'][:2000]
            prompt = f"<|bos|><|system|>{sys}<|user|>CONTEXT:\n{ctx}\n\nQuestion: Summarize the key points.<|assistant|>"
            target = ctx[:300]
            ex.append(prompt+target+"<|eos|>")
        return "\n".join(ex)

    def token_chunks(self, tok, text: str, max_tokens=480, overlap=80) -> list[str]:
        ids = tok.encode(text).ids
        chunks = []
        i = 0
        while i < len(ids):
            piece = ids[i:i+max_tokens]
            chunks.append(tok.decode(piece))
            i += max(1, max_tokens - overlap)
        return chunks

    def train(self, context_length=256, epochs=2):
        os.makedirs(self.art, exist_ok=True)
        items = PDFLoader().load(self.doc)  # [{'text':..., 'source':...}]
        plain = "\n\n".join([it['text'] for it in items])
        tok = BPETokenizer(self.tokenizer_path).build_or_load(plain)
        spec = BPETokenizer(self.tokenizer_path).get_ids(tok)
        chat = self.make_chat_corpus(items)
        corpus = plain + "\n\n" + chat

        ids = tok.encode(corpus).ids
        n = int(0.9*len(ids))
        ds_train = TextDataset(ids[:n], context_length, spec['<|pad|>'])
        ds_val = TextDataset(ids[n:], context_length, spec['<|pad|>'])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=64)

        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        model = SmallLanguageModel(vocab_size=tok.get_vocab_size(), context_length=context_length, pad_id=spec['<|pad|>']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        for e in range(epochs):
            model.train()
            for x,y in dl_train:
                x,y=x.to(device),y.to(device)
                _,loss=model(x,y)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            model.eval(); v=[]
            with torch.no_grad():
                for x,y in dl_val:
                    x,y=x.to(device),y.to(device)
                    _,l=model(x,y); v.append(l.item())
            print(f"epoch {e+1} val_ppl {math.exp(float(np.mean(v))):.2f}")

        torch.save({'state_dict': model.state_dict(), 'cfg': {'vocab_size': tok.get_vocab_size(), 'context_length': context_length, 'pad_id': spec['<|pad|>']}, 'tokenizer_path': self.tokenizer_path, 'special_ids': spec}, self.model_path)

        chunks_with_src = []
        for it in items:
            for ch in self.token_chunks(tok, it['text']):
                chunks_with_src.append({'text': ch, 'source': it['source']})
        retr = TfidfRetriever(self.retriever_path)
        retr.build_or_load(chunks_with_src)