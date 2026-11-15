"""Tests for Game Dialogue Assistant milestones."""

import pytest
from app.core import milestone_1, to_sentences, to_tokens


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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
  raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement dialogue generation logic using the model.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
  raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Design a CLI where users can input a starting phrase and generate dialogue.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
  raise NotImplementedError("write the test for milestone 4")