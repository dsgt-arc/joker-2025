# DS@GT ARC · CLEF 2025 JOKER Task 2 — Pun Translation

Georgia Tech's DS@GT ARC submission to [Task 2 (pun translation)](https://www.joker-project.com/clef-2025/) of the [CLEF 2025 JOKER lab](https://clef2025.clef-initiative.eu/index.php?page=Pages/Labs/JOKER.html): translating English puns into French.

Sentence-transformer embeddings and a FAISS index retrieve semantically related candidate words for a source pun ([`src/embeddings.py`](src/embeddings.py), [`src/preprocessor.py`](src/preprocessor.py)). An LLM generates contrastive non-pun rewrites of each source pun, used to isolate what specifically makes the sentence funny ([`src/contrastive_learning.py`](src/contrastive_learning.py)). An LLM generator then produces homonym-based French pun candidates ([`src/generator.py`](src/generator.py)), and an LLM discriminator rates each candidate's equivalence to the source on a 3-point scale — full, partial, or non-equivalence of meaning and humor ([`src/discriminator.py`](src/discriminator.py)).

This is the predecessor to [DS@GT ARC's 2026 JOKER Task 2 system](https://github.com/dsgt-arc/joker-2026), which replaced the FAISS/embeddings retrieval here with a purpose-trained phonetic embedding model and a much larger French expression bank.

Full method is in the working notes: *Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings for CLEF JOKER 2025 Task 2* (Taylor, Herbert, Sana — see [Citation](#citation)). DS@GT ARC's two submitted runs placed first and second in the official human evaluation.

## How the pipeline works

1. **Preprocessing** — identify the pun word, its two meanings, and whether it's homographic or homophonic; translate context via Google Translate; retrieve semantically similar candidate words ([`src/preprocessor.py`](src/preprocessor.py)).
2. **Contrastive example generation** — an LLM rewrites each pun as a non-punny sentence of similar length/content, iteratively re-checked by a second LLM call until it verifiably contains no wordplay ([`src/contrastive_learning.py`](src/contrastive_learning.py)).
3. **Generation** — an LLM generates a French pun built around a homonym whose two meanings span the original context and an idiomatic phrase ([`src/generator.py`](src/generator.py)).
4. **Discrimination** — an LLM rates whether the generated pun preserves the source's literal meaning, contextual meaning, and humor ([`src/discriminator.py`](src/discriminator.py)).
5. **Evaluation** — pun-location accuracy/precision/recall/F1 against manually annotated data, plus embedding-based similarity metrics ([`src/evaluator.py`](src/evaluator.py)).

See [`docs/overview.md`](docs/overview.md) for the original step-by-step design notes.

## Repository structure

```
root/
├── src/              # pipeline: preprocessor, embeddings, contrastive_learning, generator, discriminator, evaluator
├── notebooks/eda/    # exploratory notebooks (translation ranking, chain-of-thought prompting, BiLSTM phonetic embeddings)
├── data/             # JOKER task data, embedding matrix, converted phrases
├── scripts/          # sweep-logs.sh — moves slurm Report* files into logs/
├── docs/             # design notes
├── tests/
└── config.ini        # model aliases + data paths read by src/config.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install faiss-cpu numpy pandas scikit-learn sentence-transformers torch tqdm ipykernel  # see pyproject.toml
```

Set the API keys `src/config.py` reads from the environment: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY` (as needed for the models configured in [`config.ini`](config.ini)).

Scripts are run from inside `src/`, e.g. `cd src && python generator.py`.

## Citation

```bibtex
@inproceedings{taylor2025joker,
  title     = {Pun Intended: Multi-Agent Translation of Wordplay with Contrastive Learning and Phonetic-Semantic Embeddings for {CLEF} {JOKER} 2025 Task 2},
  author    = {Taylor, Russell and Herbert, Ben and Sana, M.},
  booktitle = {Working Notes of CLEF 2025 -- Conference and Labs of the Evaluation Forum},
  volume    = {4038},
  series    = {CEUR Workshop Proceedings},
  publisher = {CEUR-WS.org},
  year      = {2025},
  url       = {https://ceur-ws.org/Vol-4038/paper_229.pdf}
}
```

## Team

Russell Taylor, Ben Herbert, M. Sana — Georgia Institute of Technology, [DS@GT ARC](https://github.com/dsgt-arc) CLEF competition group.

## License

[MIT](LICENSE)
