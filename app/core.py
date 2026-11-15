"""Game Dialogue Assistant - Core ML and N-Gram Logic."""

import argparse
import random
import re
from collections import Counter, defaultdict

START_TOKEN = "<s>"
END_TOKEN = "</s>"

GAME_DIALOGUE_CORPUS = """
Greetings traveler, welcome to our humble village.
Beware the dark forest where wolves and shadow beasts roam at night.
Halt! Who goes there? State your business or face the city guard.
Take this ancient map and venture deep into the forgotten dungeon ruins.
I have rare health potions, enchanted spell rings, and sharp steel blades for sale.
The dragon has awakened from its slumber beneath the fiery mountain peak.
May the light of the sacred blade guide your path through the shadow realm.
Honor and glory await those brave enough to confront the skeleton king.
Speak with the village elder before you cross the misty river bridge.
Keep your weapon sheathed inside the tavern walls or the barkeep will call the guards.
None shall pass into the royal castle without the king's official seal.
The ancient ruins hold secrets of powerful magic long forgotten by mankind.
Rest by the campfire and regain your strength before tomorrow's perilous journey.
"""

_QUOTE_TRANSLATION = str.maketrans({
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
})

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|(?<=[.!?])[\"')\]]*$")
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def normalize(text: str) -> str:
  """Normalizes typographic quotes and whitespace."""
  return re.sub(r"\s+", " ", text.translate(_QUOTE_TRANSLATION)).strip()


def to_sentences(text: str) -> list[str]:
  """Splits raw dialogue text into clean sentences.

  Only splits on terminal punctuation followed by whitespace or end of text,
  so values like "3.5" and ellipses stay intact.
  """
  parts = _SENTENCE_END.split(normalize(text))
  cleaned = (p.strip(" \"'.!?") for p in parts if p)
  return [c for c in cleaned if c]


def to_tokens(sentence: str) -> list[str]:
  """Extracts lowercase words (Unicode letters and digits) with inner apostrophes."""
  return _TOKEN.findall(normalize(sentence).lower())


# --- Milestone 1: Collect and preprocess a corpus of game dialogues.
def milestone_1(raw_corpus: str | None = None, n: int = 2) -> list[list[str]]:
  """Preprocesses the corpus into tokenized sentences.

  Each sentence is padded with n-1 start tokens and one end token so that the
  first real word always has a full n-gram context.
  """
  if n < 1:
    raise ValueError("n must be at least 1.")

  corpus_text = raw_corpus if raw_corpus is not None else GAME_DIALOGUE_CORPUS
  if not corpus_text or not corpus_text.strip():
    raise ValueError("Corpus is empty.")

  preprocessed = []
  for sentence in to_sentences(corpus_text):
    tokens = to_tokens(sentence)
    if tokens:
      preprocessed.append([START_TOKEN] * (n - 1) + tokens + [END_TOKEN])

  if not preprocessed:
    raise ValueError("Corpus contains no usable sentences.")
  return preprocessed


def build_ngrams(token_sents, n):
  """Maps each (n-1)-word context to a Counter of the words that follow it."""
  follow = defaultdict(Counter)
  for tokens in token_sents:
    for i in range(len(tokens) - n + 1):
      follow[tuple(tokens[i : i + n - 1])][tokens[i + n - 1]] += 1
  return follow


class GameNgramLM:
  """N-gram language model with add-k smoothing."""

  def __init__(self, n=2, k=0.1):
    if n < 1:
      raise ValueError("n must be at least 1.")
    if k <= 0:
      raise ValueError("k must be positive.")
    self.n = n
    self.k = k
    self.counts = {}
    self.totals = {}
    self.vocab = set()

  def fit(self, token_sents):
    """Fits on padded token lists, as returned by milestone_1(n=self.n)."""
    self.counts = dict(build_ngrams(token_sents, self.n))
    self.totals = {ctx: sum(c.values()) for ctx, c in self.counts.items()}
    self.vocab = {t for sent in token_sents for t in sent if t != START_TOKEN}
    return self

  def _context(self, context):
    context = tuple(context)
    if self.n == 1:
      return ()
    context = context[-(self.n - 1):]
    return (START_TOKEN,) * (self.n - 1 - len(context)) + context

  def prob(self, context, word):
    """Smoothed probability: (count + k) / (total + k * |V|)."""
    ctx = self._context(context)
    count = self.counts.get(ctx, {}).get(word, 0)
    total = self.totals.get(ctx, 0)
    return (count + self.k) / (total + self.k * len(self.vocab))


# --- Milestone 2: Build the n-gram language model with smoothing.
def milestone_2(n=2, k=0.1, corpus_text=None):
  """Builds and fits a smoothed n-gram language model on the game corpus."""
  token_sents = milestone_1(corpus_text, n=n)
  return GameNgramLM(n=n, k=k).fit(token_sents)


def detokenize(tokens):
  """Turns tokens back into a readable sentence."""
  words = [t for t in tokens if t not in (START_TOKEN, END_TOKEN)]
  if not words:
    return ""
  text = " ".join(words)
  text = re.sub(r"\bi\b", "I", text)
  text = text[0].upper() + text[1:]
  if text[-1] not in ".!?":
    text += "."
  return text


def _sample(model, context, rng, temperature, top_k):
  """Samples the next token from the model's smoothed distribution."""
  words = sorted(model.vocab)
  weights = [model.prob(context, w) ** (1.0 / temperature) for w in words]

  if top_k is not None and top_k < len(words):
    ranked = sorted(range(len(words)), key=lambda i: (-weights[i], words[i]))
    keep = set(ranked[:top_k])
    weights = [w if i in keep else 0.0 for i, w in enumerate(weights)]

  return rng.choices(words, weights=weights, k=1)[0]


def generate_dialogue(model, start_phrase="", max_words=20, temperature=1.0,
                      top_k=None, seed=None):
  """Generates one line of NPC dialogue, continuing from an optional phrase."""
  if max_words < 1:
    raise ValueError("max_words must be at least 1.")
  if temperature <= 0:
    raise ValueError("temperature must be positive.")
  if top_k is not None and top_k < 1:
    raise ValueError("top_k must be at least 1.")
  if not model.vocab:
    raise ValueError("Model has not been fitted.")

  rng = random.Random(seed)
  tokens = to_tokens(start_phrase)
  generated = 0

  while generated < max_words:
    context = [START_TOKEN] * (model.n - 1) + tokens
    nxt = _sample(model, context, rng, temperature, top_k)
    if nxt == END_TOKEN:
      break
    tokens.append(nxt)
    generated += 1

  return detokenize(tokens)


# --- Milestone 3: Implement dialogue generation logic using the model.
def milestone_3(prompt="greetings traveler", max_tokens=15, seed=42, model=None,
                temperature=1.0, top_k=None):
  """Generates dialogue using the fitted n-gram model."""
  lm = model if model is not None else milestone_2(k=0.001)
  return generate_dialogue(lm, prompt, max_tokens, temperature, top_k, seed)

# --- Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
def build_parser():
  parser = argparse.ArgumentParser(
      description="Generate RPG NPC dialogue with an n-gram language model.")
  parser.add_argument("-p", "--phrase", default=None,
                      help="starting phrase; omit for interactive mode")
  parser.add_argument("-n", "--order", type=int, default=2, help="n-gram order")
  parser.add_argument("-k", "--smoothing", type=float, default=0.001,
                      help="add-k smoothing constant")
  parser.add_argument("-m", "--max-words", type=int, default=20)
  parser.add_argument("-t", "--temperature", type=float, default=1.0)
  parser.add_argument("--top-k", type=int, default=None)
  parser.add_argument("-c", "--count", type=int, default=1,
                      help="lines to generate per phrase")
  parser.add_argument("--seed", type=int, default=None)
  return parser
# --- Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
def milestone_4(argv=None, input_fn=input, output_fn=print):
  """Runs the CLI and returns an exit code."""
  args = build_parser().parse_args(argv)

  try:
    model = milestone_2(n=args.order, k=args.smoothing)
  except ValueError as err:
    output_fn(f"Error: {err}")
    return 1

  def generate(phrase, offset):
    for i in range(args.count):
      seed = None if args.seed is None else args.seed + offset + i
      yield generate_dialogue(model, phrase, args.max_words, args.temperature,
                              args.top_k, seed)

  try:
    if args.phrase is not None:
      for line in generate(args.phrase, 0):
        output_fn(line)
      return 0

    output_fn("NPC Dialogue Generator. Type a starting phrase, or 'quit' to exit.")
    turn = 0
    while True:
      try:
        phrase = input_fn("> ").strip()
      except EOFError:
        break
      if phrase.lower() in ("quit", "exit", "q"):
        break
      for line in generate(phrase, turn * args.count):
        output_fn(line)
      turn += 1
    return 0
  except ValueError as err:
    output_fn(f"Error: {err}")
    return 1


if __name__ == "__main__":
  raise SystemExit(milestone_4())