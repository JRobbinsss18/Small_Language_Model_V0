import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self, path: str, vocab_size: int = 4096):
        self.path = path
        self.vocab_size = vocab_size
        self.special = ["<|bos|>","<|eos|>","<|system|>","<|user|>","<|assistant|>","<|pad|>"]
    def _new(self):
        tok = Tokenizer(BPE(unk_token=None))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = ByteLevelDecoder()
        return tok
    def build_or_load(self, corpus: str | None) -> Tokenizer:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if os.path.exists(self.path):
            tok = Tokenizer.from_file(self.path)
            if not isinstance(tok.pre_tokenizer, ByteLevel):
                tok.pre_tokenizer = ByteLevel()
            tok.decoder = ByteLevelDecoder()
            v = tok.get_vocab()
            missing = [s for s in self.special if s not in v]
            if missing:
                tok.add_special_tokens(missing)
                tok.save(self.path)
            return tok
        if corpus is None:
            raise RuntimeError('corpus required to build tokenizer')
        tok = self._new()
        trainer = BpeTrainer(vocab_size=self.vocab_size - len(self.special), special_tokens=self.special)
        tok.train_from_iterator([corpus], trainer)
        tok.save(self.path)
        return tok
    def get_ids(self, tok: Tokenizer) -> dict:
        v = tok.get_vocab()
        missing = [s for s in self.special if s not in v]
        if missing:
            tok.add_special_tokens(missing)
            tok.save(self.path)
            v = tok.get_vocab()
        return {s: v[s] for s in self.special}