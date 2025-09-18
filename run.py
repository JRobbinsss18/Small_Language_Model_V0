import argparse
from trainer import Trainer
from rag_pipeline import RAGPipeline

class CLI:
    def run(self):
        p = argparse.ArgumentParser()
        p.add_argument('--train', action='store_true')
        p.add_argument('--context_length', type=int, default=256)
        p.add_argument('--epochs', type=int, default=2)
        p.add_argument('--ask', type=str, default=None)
        p.add_argument('--topk', type=int, default=5)
        args = p.parse_args()
        if args.train:
            Trainer().train(context_length=args.context_length, epochs=args.epochs)
        if args.ask:
            print(RAGPipeline().answer(args.ask, k=args.topk))

if __name__ == '__main__':
    CLI().run()