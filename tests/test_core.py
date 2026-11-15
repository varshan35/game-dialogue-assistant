"""Tests for Game Dialogue Assistant milestones."""

import pytest
from app.core import (
    detokenize,
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    to_sentences,
    to_tokens,
)


# Milestone 1: Collect and preprocess a corpus of game dialogues.
def test_milestone_1():
  """Tests that the game corpus is loaded and properly preprocessed."""
  corpus_tokens = milestone_1()

  assert isinstance(corpus_tokens, list)
  assert len(corpus_tokens) > 5

  for sentence in corpus_tokens:
    assert sentence[0] == "<s>"
    assert sentence[-1] == "</s>"
    assert len(sentence) > 2
    assert all(isinstance(t, str) for t in sentence)
    assert all(t == t.lower() for t in sentence)

  first_sentence = corpus_tokens[0]
  assert "greetings" in first_sentence
  assert "traveler" in first_sentence


def test_milestone_1_sentence_count():
  """Tests that every sentence in the corpus is kept (15, since one line has three)."""
  assert len(milestone_1()) == 15


def test_milestone_1_custom_corpus():
  """Tests that a custom corpus overrides the default one."""
  result = milestone_1("Hello there. Farewell!")
  assert result == [
      ["<s>", "hello", "there", "</s>"],
      ["<s>", "farewell", "</s>"],
  ]


def test_milestone_1_ngram_padding():
  """Tests that n-1 start tokens are added for higher-order models."""
  result = milestone_1("Hello there.", n=3)
  assert result == [["<s>", "<s>", "hello", "there", "</s>"]]


def test_milestone_1_empty_corpus():
  """Tests that empty or punctuation-only input raises an error."""
  with pytest.raises(ValueError):
    milestone_1("")
  with pytest.raises(ValueError):
    milestone_1("   \n  ")
  with pytest.raises(ValueError):
    milestone_1("... !!! ???")


def test_milestone_1_invalid_n():
  """Tests that n below 1 raises an error."""
  with pytest.raises(ValueError):
    milestone_1("Hello.", n=0)


def test_to_sentences_basic():
  """Tests basic sentence splitting on . ! and ?."""
  assert to_sentences("Halt! Who goes there? Speak.") == [
      "Halt",
      "Who goes there",
      "Speak",
  ]


def test_to_sentences_keeps_decimals():
  """Tests that decimals do not split a sentence."""
  assert to_sentences("It costs 3.5 gold.") == ["It costs 3.5 gold"]


def test_to_sentences_quoted_dialogue():
  """Tests that quotes around dialogue do not create stray fragments."""
  assert to_sentences('"Halt!" he said.') == ["Halt", "he said"]


def test_to_sentences_empty():
  """Tests that blank input yields no sentences."""
  assert to_sentences("") == []
  assert to_sentences("  ... ") == []


def test_to_tokens_lowercases():
  """Tests that tokens are lowercased."""
  assert to_tokens("Greetings Traveler") == ["greetings", "traveler"]


def test_to_tokens_apostrophes():
  """Tests straight and curly apostrophes both stay inside words."""
  assert to_tokens("the king's seal") == ["the", "king's", "seal"]
  assert to_tokens("the king\u2019s seal") == ["the", "king's", "seal"]


def test_to_tokens_unicode_and_digits():
  """Tests that accented letters and digits are preserved."""
  assert to_tokens("Caf\u00e9 level 5") == ["caf\u00e9", "level", "5"]


def test_to_tokens_strips_punctuation():
  """Tests that punctuation is not part of tokens."""
  assert to_tokens("Beware, traveler!") == ["beware", "traveler"]


# Milestone 2: Build the n-gram language model with smoothing.
def test_milestone_2():
  """Tests vocabulary, probability axioms, and add-k smoothing."""
  model = milestone_2(n=2, k=0.1)

  assert len(model.vocab) > 10
  assert "<s>" not in model.vocab
  assert "</s>" in model.vocab

  for ctx in [("greetings",), ("the",), ("<s>",), ("zzz",)]:
    assert abs(sum(model.prob(ctx, w) for w in model.vocab) - 1.0) < 1e-9

  assert model.prob(("greetings",), "nonexistent_word_xyz") > 0.0


def test_milestone_2_seen_beats_unseen():
  """Tests that an observed continuation is likelier than an unobserved one."""
  model = milestone_2(n=2, k=0.1)
  assert model.prob(("greetings",), "traveler") > model.prob(("greetings",), "dragon")


def test_milestone_2_exact_probability():
  """Tests the add-k formula by hand: (count + k) / (total + k * |V|)."""
  model = milestone_2(n=2, k=0.1)
  expected = (1 + 0.1) / (1 + 0.1 * len(model.vocab))
  assert model.prob(("greetings",), "traveler") == pytest.approx(expected)


def test_milestone_2_unseen_context_is_uniform():
  """Tests that an unseen context falls back to a uniform distribution."""
  model = milestone_2(n=2, k=0.1)
  assert model.prob(("zzz",), "dragon") == pytest.approx(1 / len(model.vocab))


def test_milestone_2_trigram_and_short_context():
  """Tests that short contexts are left-padded with <s>."""
  model = milestone_2(n=3, k=0.1)
  assert model.prob(("greetings",), "traveler") == model.prob(("<s>", "greetings"), "traveler")
  assert abs(sum(model.prob(("<s>", "greetings"), w) for w in model.vocab) - 1.0) < 1e-9


def test_milestone_2_invalid_params():
  """Tests that bad n or k raises an error."""
  with pytest.raises(ValueError):
    milestone_2(k=0)
  with pytest.raises(ValueError):
    milestone_2(n=0)


# Milestone 3: Implement dialogue generation logic using the model.
def test_detokenize():
  """Tests capitalization, the pronoun I, and closing punctuation."""
  assert detokenize(["<s>", "i", "am", "here", "</s>"]) == "I am here."
  assert detokenize(["well", "i'm", "ready!"]) == "Well I'm ready!"
  assert detokenize(["<s>", "</s>"]) == ""


def test_milestone_3():
  """Tests that generation returns a well-formed line starting with the phrase."""
  model = milestone_2(n=2, k=0.001)
  line = milestone_3(model, "the dragon", max_words=15, seed=1)

  assert isinstance(line, str)
  assert line.startswith("The dragon")
  assert line[-1] in ".!?"
  assert len(to_tokens(line)) <= 2 + 15


def test_milestone_3_is_reproducible():
  """Tests that the same seed gives the same line."""
  model = milestone_2(n=2, k=0.001)
  a = milestone_3(model, "beware the", seed=42)
  b = milestone_3(model, "beware the", seed=42)
  assert a == b


def test_milestone_3_respects_max_words():
  """Tests that no more than max_words new words are added."""
  model = milestone_2(n=2, k=0.1)
  for seed in range(10):
    line = milestone_3(model, "", max_words=5, seed=seed)
    assert len(to_tokens(line)) <= 5


def test_milestone_3_top_k_one_is_greedy():
  """Tests that top_k=1 always picks the most likely word."""
  model = milestone_2(n=2, k=0.001)
  a = milestone_3(model, "greetings", top_k=1, seed=1)
  b = milestone_3(model, "greetings", top_k=1, seed=99)
  assert a == b
  assert a.startswith("Greetings traveler")


def test_milestone_3_no_start_phrase():
  """Tests generation from nothing."""
  model = milestone_2(n=2, k=0.001)
  assert len(milestone_3(model, "", seed=3)) > 0


def test_milestone_3_invalid_params():
  """Tests that bad generation settings raise errors."""
  model = milestone_2()
  with pytest.raises(ValueError):
    milestone_3(model, max_words=0)
  with pytest.raises(ValueError):
    milestone_3(model, temperature=0)
  with pytest.raises(ValueError):
    milestone_3(model, top_k=0)


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
def test_milestone_4_single_phrase():
  """Tests one-shot mode with a starting phrase."""
  out = []
  code = milestone_4(["-p", "the dragon", "-c", "3", "--seed", "1"], output_fn=out.append)
  assert code == 0
  assert len(out) == 3
  assert all(line.startswith("The dragon") for line in out)


def test_milestone_4_interactive():
  """Tests interactive mode and quitting."""
  out = []
  answers = iter(["greetings", "beware", "quit"])
  code = milestone_4(["--seed", "1"], input_fn=lambda _: next(answers),
                     output_fn=out.append)
  assert code == 0
  assert len(out) == 3
  assert out[1].startswith("Greetings")
  assert out[2].startswith("Beware")


def test_milestone_4_eof_exits_cleanly():
  """Tests that end of input ends interactive mode without an error."""
  def raise_eof(_):
    raise EOFError

  assert milestone_4([], input_fn=raise_eof, output_fn=lambda _: None) == 0


def test_milestone_4_bad_args_return_error():
  """Tests that invalid settings print an error and return 1."""
  out = []
  assert milestone_4(["-k", "0", "-p", "hi"], output_fn=out.append) == 1
  assert out and out[0].startswith("Error")
  out = []
  assert milestone_4(["-m", "0", "-p", "hi"], output_fn=out.append) == 1