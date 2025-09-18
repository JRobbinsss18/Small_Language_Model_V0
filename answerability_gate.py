import re
from typing import List, Union, Dict

class AnswerabilityGate:
    def is_answerable(self, query: str, docs: List[Union[str, Dict[str, str]]]) -> bool:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
        if not terms:
            return False

        def text_of(d):
            return d["text"] if isinstance(d, dict) else d

        hits = 0
        for t in terms:
            if any(t in text_of(d).lower() for d in docs):
                hits += 1
        return hits / max(1, len(terms)) > 0.3