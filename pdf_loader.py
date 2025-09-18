import os, re
from typing import List, Dict
from pypdf import PdfReader

class PDFLoader:
    def load(self, root: str) -> List[Dict[str, str]]:
        out = []
        for dp, _, fns in os.walk(root):
            for fn in sorted(fns):
                if not fn.lower().endswith('.pdf'):
                    continue
                path = os.path.join(dp, fn)
                try:
                    r = PdfReader(path)
                    pages = [p.extract_text() or '' for p in r.pages]
                    raw = '\n'.join(pages)
                except Exception:
                    continue
                txt = self.clean(raw)
                if txt.strip():
                    out.append({'text': txt, 'source': fn})
        if not out:
            raise RuntimeError('no pdf text')
        return out
    def clean(self, x: str) -> str:
        x = x.replace('\x00', ' ')
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"\|\s*Page \d+\s*\|", " ", x)
        x = re.sub(r"\s+-\s+\d+\s+$", " ", x)
        return x.strip()