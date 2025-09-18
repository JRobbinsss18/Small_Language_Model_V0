# Small Language Model with RAG and PDF Support

This project is based on the small language model tutorial by [Rajasami on Medium](https://medium.com/@rajasami408/building-a-small-language-model-from-scratch-a-practical-guide-to-domain-specific-ai-59539131437f), which walks through building a small language model from scratch.  

We have extended that base with severalg improvements:

- **Document citation** — answers show source PDF names.
- **Cleaner, less rambling responses** — a compressor step selects concise, relevant sentences to reduce clutter.
- **Answerability gate** — adds a point at which we say it could not be found.
- **Automatic retriever upgrades** — old TF-IDF indexes are auto-rebuilt to preserve filenames as sources.

---

## Setup

1. Place your domain-specific PDFs in the `Documents/` folder.
2. Install requirements:
   ```bash
   pip install -r requirements.txt   
3. Train model if needed (example below, not the required line)
    ```
    python run.py --train --epochs 2 --context_length 256 
4. Ask model questions (example below, not the required line)
    ```
    python run.py --ask "What is most important about nutrition" --topk 2