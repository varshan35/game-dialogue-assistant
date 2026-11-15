# Game Dialogue Assistant

A command-line tool that helps game designers prototype NPC dialogue quickly and consistently. It trains a smoothed n-gram language model on a corpus of RPG-style dialogue and generates new lines that continue from any starting phrase.

## Overview

Writing dozens of NPC lines in a consistent voice is slow. The Game Dialogue Assistant learns the vocabulary and phrasing of a dialogue corpus, then proposes new lines in the same style. Designers can steer the output with a starting phrase, control how adventurous it sounds, and reproduce any result with a seed.

**Highlights**

- Unicode-aware preprocessing that handles curly quotes, accents, digits and decimals
- Configurable n-gram order with add-k smoothing
- Temperature and top-k sampling to trade consistency against variety
- Reproducible output through random seeds
- One-shot and interactive command-line modes
- Fully tested, with no external runtime dependencies

## Getting Started

### Requirements

- Python 3.10 or newer
- git

### Installation

```bash
git clone <repository-url>
cd game-dialogue-assistant
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generate a line

```bash
python -m app.core -p "the dragon" --seed 1
```

```text
The dragon has awakened from its slumber beneath the shadow beasts fiery mountain peak.
```

### Generate several alternatives

```bash
python -m app.core -p "keep your" -c 2 --top-k 3 --seed 7
```

```text
Keep your business or the ancient ruins.
Keep your business or face the ancient map and glory await those brave enough to our humble village.
```

### Interactive mode

Run without `-p` to type phrases one after another. Enter `quit`, `exit` or `q` (or press Ctrl+D) to leave.

```bash
python -m app.core
```

```text
NPC Dialogue Generator. Type a starting phrase, or 'quit' to exit.
> greetings
> quit
```

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `-p`, `--phrase` | none | Starting phrase. Omit it to enter interactive mode. |
| `-n`, `--order` | `2` | N-gram order. 2 is a bigram model, 3 a trigram model. |
| `-k`, `--smoothing` | `0.001` | Add-k smoothing constant. Must be positive. |
| `-m`, `--max-words` | `20` | Maximum number of words to add after the phrase. |
| `-t`, `--temperature` | `1.0` | Below 1 gives safer text, above 1 gives wilder text. |
| `--top-k` | none | Sample only from the k most likely words. `1` is fully greedy. |
| `-c`, `--count` | `1` | Number of lines to generate per phrase. |
| `--seed` | none | Seed for reproducible output. |

Invalid settings print `Error: ...` and exit with status 1 instead of a traceback.

### Use it as a library

```python
from app.core import milestone_2, generate_dialogue

model = milestone_2(n=2, k=0.001)
print(generate_dialogue(model, "beware the", max_words=15, temperature=0.8, seed=4))
```

To train on your own script, pass your text as `corpus_text`:

```python
model = milestone_2(n=3, k=0.001, corpus_text=open("my_npc_script.txt").read())
```

## How It Works

1. **Preprocess.** Text is normalized, split into sentences, lowercased and tokenized. Each sentence is padded with `n-1` start tokens (`<s>`) and one end token (`</s>`).
2. **Model.** Counts of every `(n-1)`-word context and the words that follow it are collected. Probabilities use add-k smoothing so that unseen continuations are never impossible:

   ```text
   P(word | context) = (count(context, word) + k) / (count(context) + k * |V|)
   ```

3. **Generate.** Starting from the phrase, the model repeatedly samples the next word from this distribution, optionally sharpened by temperature and restricted by top-k, until it draws `</s>` or reaches the word limit.
4. **Format.** Tokens are rejoined, the first letter is capitalized, the pronoun "I" is restored and a closing period is added.

## Project Structure

```text
.
├── app/
│   └── core.py          # Preprocessing, language model, generation and CLI
├── tests/
│   ├── test_core.py     # Unit tests for all four milestones
│   └── test_smoke.py    # Plumbing checks
├── pytest.ini
├── requirements.txt
└── README.md
```

## Testing

```bash
python -m pytest
```

The suite covers tokenization edge cases, probability axioms (distributions sum to 1), the exact smoothing formula, reproducibility under a fixed seed, parameter validation and both CLI modes.

## Design Notes and Limitations

- **Small corpus.** The bundled corpus has only 15 sentences, so a bigram model often stitches together fragments of training lines and a trigram model mostly repeats them. The best way to improve quality is to add more dialogue to `GAME_DIALOGUE_CORPUS` or supply your own script.
- **Smoothing and quality.** With a small corpus, a large `k` such as 0.1 spreads most probability onto words never seen after a context, producing disconnected text. The CLI therefore defaults to `k=0.001`.
- **Out-of-vocabulary words.** A word outside the training vocabulary receives a small non-zero probability, so the full distribution sums to slightly more than 1. This does not affect generation, but an `<unk>` token would be needed for perplexity scoring.
- **Abbreviations.** Sentence splitting treats "Mr." as a sentence end. Decimals such as `3.5` are handled correctly.

## Roadmap

- [x] Corpus collection and preprocessing
- [x] N-gram language model with add-k smoothing
- [x] Dialogue generation with temperature and top-k sampling
- [x] Command-line interface with one-shot and interactive modes
- [x] User-specified output length (`--max-words`)
- [x] Multiple alternative suggestions per phrase (`--count`)
- [ ] Save and load trained models
- [ ] `<unk>` token for out-of-vocabulary words
- [ ] Per-character corpora to keep each NPC's voice distinct

## Security

No API keys or secrets are needed. If you add settings that do need them, keep them in a git-ignored `.env` file and never commit them or paste them into code.
