import re
from typing import List, Dict

class Compressor:
    def _sentences(self, text: str) -> List[str]:
        s = re.split(r'(?<=[\.\?\!])\s+', text.strip())
        s = [t.strip() for t in s if 30 <= len(t.strip()) <= 220]
        return s

    def _tokenize(self, x: str) -> List[str]:
        return [t for t in re.findall(r"\w+", x.lower()) if len(t) > 2]

    def _sim(self, a: str, b: str) -> float:
        A, B = set(self._tokenize(a)), set(self._tokenize(b))
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)

    def extract(self, question: str, docs: List[Dict[str,str]], max_points: int = 5) -> List[Dict[str,str]]:
        q_terms = self._tokenize(question)
        boost = set()
        if any(t in q_terms for t in ['alcohol','drink','drinking','beer','wine','ethanol','nightcap']):
            boost |= {'alcohol','drink','drinking','beer','wine','ethanol','nightcap'}
        if any(t in q_terms for t in ['sleep','insomnia','rest','restful','nap']):
            boost |= {'sleep','insomnia','rest','nap','rem','nonrem','circadian'}

        cands = []
        for d in docs:
            for s in self._sentences(d['text']):
                toks = set(self._tokenize(s))
                score = sum(1 for t in q_terms if t in toks)
                score += 2 * sum(1 for t in boost if t in toks)
                if score > 0:
                    cands.append({'text': s, 'source': d['source'], 'score': score})

        cands.sort(key=lambda x: (-x['score'], len(x['text'])))

        out = []
        for c in cands:
            if len(out) >= max_points: break
            if any(self._sim(c['text'], z['text']) > 0.6 for z in out): continue
            out.append({'text': c['text'], 'source': c['source']})
        return out
