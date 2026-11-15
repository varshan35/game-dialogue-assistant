# Game Dialogue Assistant

The Game Dialogue Assistant helps game designers quickly generate consistent dialogue for NPCs. By leveraging a next-word predictor trained on game scripts, this tool ensures character voice consistency and speeds up the prototyping process. It demonstrates the power of language models in creative applications.

## What you are building

**Goal.** You will build a command-line tool that assists game designers in generating dialogue snippets for NPCs, ensuring character consistency and rapid prototyping.

**Done when.** The project is complete when the CLI tool can take a starting phrase from a user and generate coherent dialogue snippets using an n-gram language model trained on game dialogue.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Collect and preprocess a corpus of game dialogues.**  
      Hint: Text Preprocessing and Tokenization
- [ ] **2. Build the n-gram language model with smoothing.**  
      Hint: N-gram Model Construction and Smoothing
- [ ] **3. Implement dialogue generation logic using the model.**  
      Hint: Text Generation with N-grams
- [ ] **4. Design a CLI where users can input a starting phrase and generate dialogue.**  
      Hint: Command-Line Interface (CLI) Design

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the desired length of the generated dialogue.
- Offer multiple alternative dialogue suggestions for a given starting phrase.
- Implement functionality to save and load trained language models.
