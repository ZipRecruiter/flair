import pytest

import flair
from flair.data import Sentence, Token
from flair.splitter import (
    NewlineSentenceSplitter,
    NoSentenceSplitter,
    SciSpacySentenceSplitter,
    SegtokSentenceSplitter,
    SpacySentenceSplitter,
    TagSentenceSplitter,
)
from flair.tokenization import (
    JapaneseTokenizer,
    SciSpacyTokenizer,
    SegtokTokenizer,
    SpaceTokenizer,
    SpacyTokenizer,
    TokenizerWrapper,
    StaccatoTokenizer,
)
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings


def test_create_sentence_on_empty_string():
    sentence: Sentence = Sentence("")
    assert len(sentence.tokens) == 0


def test_create_sentence_with_newline():
    sentence: Sentence = Sentence(["I", "\t", "ich", "\n", "you", "\t", "du", "\n"])
    assert len(sentence.tokens) == 8
    assert sentence.tokens[3].text == "\n"

    sentence: Sentence = Sentence("I \t ich \n you \t du \n", use_tokenizer=False)
    assert len(sentence.tokens) == 8
    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[3].text == "\n"


def test_create_sentence_with_extra_whitespace():
    sentence: Sentence = Sentence("I  love Berlin .")

    assert len(sentence.tokens) == 4
    assert sentence.get_token(1).text == "I"
    assert sentence.get_token(2).text == "love"
    assert sentence.get_token(3).text == "Berlin"
    assert sentence.get_token(4).text == "."


@pytest.mark.skip(reason="Fix these issues for StaccatoTokenizer in future PR")
def test_create_sentence_difficult_encoding():
    text = "so out of the norm ❤ ️ enjoyed every moment️"
    sentence = Sentence(text)
    assert len(sentence) == 9

    text = (
        "equivalently , accumulating the logs as :( 6 ) sl = 1N ∑ t = 1Nlogp "
        "( Ll | xt \u200b , θ ) where "
        "p ( Ll | xt \u200b , θ ) represents the class probability output"
    )
    sentence = Sentence(text)
    assert len(sentence) == 37

    text = "This guy needs his own show on Discivery Channel ! ﻿"
    sentence = Sentence(text)
    assert len(sentence) == 10

    text = "n't have new vintages."
    sentence = Sentence(text, use_tokenizer=True)
    assert len(sentence) == 5


def test_create_sentence_word_by_word():
    token1: Token = Token("Munich")
    token2: Token = Token("and")
    token3: Token = Token("Berlin")
    token4: Token = Token("are")
    token5: Token = Token("nice")

    sentence: Sentence = Sentence([token1, token2, token3, token4, token5, Token("cities"), Token(".")])

    assert sentence.to_tokenized_string() == "Munich and Berlin are nice cities ."


def test_create_sentence_pretokenized():
    pretoks = ["The", "grass", "is", "green", "."]
    sent = Sentence(pretoks)
    for i, token in enumerate(sent):
        assert token.text == pretoks[i]


def test_create_sentence_without_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=False)

    assert len(sentence.tokens) == 3
    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].text == "I"
    assert sentence.tokens[1].start_position == 2
    assert sentence.tokens[1].text == "love"
    assert sentence.tokens[2].start_position == 7
    assert sentence.tokens[2].text == "Berlin."


def test_create_sentence_with_default_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=True)

    assert len(sentence.tokens) == 4
    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].text == "I"
    assert sentence.tokens[1].start_position == 2
    assert sentence.tokens[1].text == "love"
    assert sentence.tokens[2].start_position == 7
    assert sentence.tokens[2].text == "Berlin"
    assert sentence.tokens[3].start_position == 13
    assert sentence.tokens[3].text == "."


def test_create_sentence_with_segtok():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert len(sentence.tokens) == 4
    assert sentence.tokens[0].text == "I"
    assert sentence.tokens[1].text == "love"
    assert sentence.tokens[2].text == "Berlin"
    assert sentence.tokens[3].text == "."


def test_create_sentence_with_custom_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=TokenizerWrapper(no_op_tokenizer))
    assert len(sentence.tokens) == 1
    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].text == "I love Berlin."


@pytest.mark.skip(reason="SpacyTokenizer needs optional requirements, so we skip the test by default")
def test_create_sentence_with_spacy_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SpacyTokenizer("en_core_sci_sm"))

    assert len(sentence.tokens) == 4
    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].text == "I"
    assert sentence.tokens[1].start_position == 2
    assert sentence.tokens[1].text == "love"
    assert sentence.tokens[2].start_position == 7
    assert sentence.tokens[2].text == "Berlin"
    assert sentence.tokens[3].start_position == 13
    assert sentence.tokens[3].text == "."


def test_create_sentence_using_japanese_tokenizer():
    sentence: Sentence = Sentence("私はベルリンが好き", use_tokenizer=JapaneseTokenizer("janome"))

    assert len(sentence.tokens) == 5
    assert sentence.tokens[0].text == "私"
    assert sentence.tokens[1].text == "は"
    assert sentence.tokens[2].text == "ベルリン"
    assert sentence.tokens[3].text == "が"
    assert sentence.tokens[4].text == "好き"


@pytest.mark.skip(reason="SciSpacyTokenizer need optional requirements, so we skip the test by default")
def test_create_sentence_using_scispacy_tokenizer():
    sentence: Sentence = Sentence(
        "Spinal and bulbar muscular atrophy (SBMA) is an inherited motor neuron",
        use_tokenizer=SciSpacyTokenizer(),
    )

    assert len(sentence.tokens) == 13
    assert sentence.tokens[0].text == "Spinal"
    assert sentence.tokens[1].text == "and"
    assert sentence.tokens[2].text == "bulbar"
    assert sentence.tokens[3].text == "muscular"
    assert sentence.tokens[4].text == "atrophy"
    assert sentence.tokens[5].text == "("
    assert sentence.tokens[6].text == "SBMA"
    assert sentence.tokens[7].text == ")"
    assert sentence.tokens[8].text == "is"
    assert sentence.tokens[9].text == "an"
    assert sentence.tokens[10].text == "inherited"
    assert sentence.tokens[11].text == "motor"
    assert sentence.tokens[12].text == "neuron"

    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[1].start_position == 7
    assert sentence.tokens[2].start_position == 11
    assert sentence.tokens[3].start_position == 18
    assert sentence.tokens[4].start_position == 27
    assert sentence.tokens[5].start_position == 35
    assert sentence.tokens[6].start_position == 36
    assert sentence.tokens[7].start_position == 40
    assert sentence.tokens[8].start_position == 42
    assert sentence.tokens[9].start_position == 45
    assert sentence.tokens[10].start_position == 48
    assert sentence.tokens[11].start_position == 58
    assert sentence.tokens[12].start_position == 64

    assert sentence.tokens[4].whitespace_after == 1
    assert sentence.tokens[5].whitespace_after != 1
    assert sentence.tokens[6].whitespace_after != 1
    assert sentence.tokens[7].whitespace_after == 1


def test_split_text_segtok():
    segtok_splitter = SegtokSentenceSplitter()
    sentences = segtok_splitter._perform_split("I love Berlin. Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_position == 15
    assert len(sentences[1].tokens) == 6

    segtok_splitter = SegtokSentenceSplitter(tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = segtok_splitter._perform_split("I love Berlin. Berlin is a great city.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_position == 15
    assert len(sentences[1].tokens) == 1


def test_split_text_nosplit():
    no_splitter = NoSentenceSplitter()
    sentences = no_splitter._perform_split("I love Berlin")
    assert len(sentences) == 1
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 3

    no_splitter = NoSentenceSplitter(TokenizerWrapper(no_op_tokenizer))
    sentences = no_splitter._perform_split("I love Berlin")
    assert len(sentences) == 1
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 1


def test_split_text_on_tag():
    tag_splitter = TagSentenceSplitter(tag="#!")

    sentences = tag_splitter._perform_split("I love Berlin#!Me too")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 3
    assert sentences[1].start_position == 15
    assert len(sentences[1].tokens) == 2

    tag_splitter = TagSentenceSplitter(tag="#!", tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = tag_splitter._perform_split("I love Berlin#!Me too")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_position == 15
    assert len(sentences[1].tokens) == 1

    sentences = tag_splitter._perform_split("I love Berlin Me too")
    assert len(sentences) == 1

    sentences = tag_splitter._perform_split("I love Berlin#!#!Me too")
    assert len(sentences) == 2

    sentences = tag_splitter._perform_split("I love Berl#! #!inMe too")
    assert len(sentences) == 2


def test_split_text_on_newline():
    newline_splitter = NewlineSentenceSplitter()

    sentences = newline_splitter._perform_split("I love Berlin\nMe too")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 3
    assert sentences[0].start_position == 0
    assert len(sentences[1].tokens) == 2

    newline_splitter = NewlineSentenceSplitter(tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = newline_splitter._perform_split("I love Berlin\nMe too")
    assert len(sentences) == 2
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_position == 14
    assert len(sentences[1].tokens) == 1

    sentences = newline_splitter._perform_split("I love Berlin Me too")
    assert len(sentences) == 1

    sentences = newline_splitter._perform_split("I love Berlin\n\nMe too")
    assert len(sentences) == 2

    sentences = newline_splitter._perform_split("I love Berlin\n \nMe too")
    assert len(sentences) == 2


def test_split_sentence_linkage():
    splitter = SegtokSentenceSplitter()

    text = "This is a single sentence."
    sentences = splitter.split(text)

    assert len(sentences) == 1
    assert sentences[0].previous_sentence() is None
    assert sentences[0].next_sentence() is None

    text = "This is a sentence. This is another sentence. This is yet another sentence."
    sentences = splitter.split(text)

    assert len(sentences) == 3
    assert sentences[0].previous_sentence() is None
    assert sentences[0].next_sentence() == sentences[1]
    assert sentences[1].previous_sentence() == sentences[0]
    assert sentences[1].next_sentence() == sentences[2]
    assert sentences[2].previous_sentence() == sentences[1]
    assert sentences[2].next_sentence() is None


def test_split_sentence_linkage_false():
    splitter = SegtokSentenceSplitter()

    text = "This is a sentence. This is another sentence. This is yet another sentence."
    sentences = splitter.split(text, link_sentences=False)

    assert len(sentences) == 3
    assert all(s.next_sentence() is None and s.previous_sentence() is None for s in sentences)


@pytest.mark.skip(reason="SpacySentenceSplitter need optional requirements, so we skip the test by default")
def test_split_text_spacy():
    spacy_splitter = SpacySentenceSplitter("en_core_sci_sm")

    sentences = spacy_splitter._perform_split("This a sentence. And here is another one.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_position == 17
    assert len(sentences[1].tokens) == 6

    sentences = spacy_splitter._perform_split("VF inhibits something. ACE-dependent (GH+) issuses too.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_position == 23
    assert len(sentences[1].tokens) == 7

    spacy_splitter = SpacySentenceSplitter("en_core_sci_sm", tokenizer=TokenizerWrapper(no_op_tokenizer))
    sentences = spacy_splitter._perform_split("This a sentence. And here is another one.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 1
    assert sentences[1].start_position == 17
    assert len(sentences[1].tokens) == 1


@pytest.mark.skip(reason="SciSpacySentenceSplitter need optional requirements, so we skip the test by default")
def test_split_text_scispacy():
    scispacy_splitter = SciSpacySentenceSplitter()
    sentences = scispacy_splitter._perform_split("VF inhibits something. ACE-dependent (GH+) issuses too.")
    assert len(sentences) == 2
    assert sentences[0].start_position == 0
    assert len(sentences[0].tokens) == 4
    assert sentences[1].start_position == 23
    assert len(sentences[1].tokens) == 9


def test_print_sentence_tokenized():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert sentence.to_tokenized_string() == "I love Berlin ."


def test_print_original_text():
    text = ":    nation on"
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = ":    nation on"
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()

    text = "I love Berlin."
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " '
        "in einer Weise aufgetreten , die alles andere als überzeugend "
        'war " .'
    )
    sentence = Sentence(text)
    assert text == sentence.to_original_text()

    text = (
        'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " '
        "in einer Weise aufgetreten , die alles andere als überzeugend "
        'war " .'
    )
    sentence = Sentence(text, use_tokenizer=SegtokTokenizer())
    assert text == sentence.to_original_text()


def test_print_sentence_plain(tasks_base_path):
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())
    assert sentence.to_plain_string() == "I love Berlin."

    corpus = flair.datasets.NER_GERMAN_GERMEVAL(base_path=tasks_base_path)

    sentence = corpus.train[0]
    sentence.infer_space_after()
    assert (
        sentence.to_tokenized_string() == 'Schartau sagte dem " Tagesspiegel " vom Freitag , Fischer sei " in '
        "einer Weise aufgetreten , "
        'die alles andere als überzeugend war " .'
    )
    assert (
        sentence.to_plain_string() == 'Schartau sagte dem "Tagesspiegel" vom Freitag, Fischer sei "in einer '
        "Weise aufgetreten, die "
        'alles andere als überzeugend war".'
    )

    sentence = corpus.train[1]
    sentence.infer_space_after()
    assert (
        sentence.to_tokenized_string() == "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als "
        "Möbelvertreter , als er einen fliegenden Händler aus dem Libanon traf ."
    )
    assert (
        sentence.to_plain_string() == "Firmengründer Wolf Peter Bree arbeitete Anfang der siebziger Jahre als "
        "Möbelvertreter, als er einen fliegenden Händler aus dem Libanon traf."
    )


def test_infer_space_after():
    sentence: Sentence = Sentence([Token("xyz"), Token('"'), Token("abc"), Token('"')])
    sentence.infer_space_after()

    assert sentence.to_tokenized_string() == 'xyz " abc "'
    assert sentence.to_plain_string() == 'xyz "abc"'

    sentence: Sentence = Sentence('xyz " abc "')
    sentence.infer_space_after()
    assert sentence.to_tokenized_string() == 'xyz " abc "'
    assert sentence.to_plain_string() == 'xyz "abc"'


def test_sentence_get_item():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=SegtokTokenizer())

    assert sentence.get_token(1) == sentence[0]
    assert sentence.get_token(3) == sentence[2]

    with pytest.raises(IndexError):
        _ = sentence[4]


def test_token_positions_when_creating_with_tokenizer():
    sentence = Sentence("I love Berlin .", use_tokenizer=SpaceTokenizer())

    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].end_position == 1
    assert sentence.tokens[1].start_position == 2
    assert sentence.tokens[1].end_position == 6
    assert sentence.tokens[2].start_position == 7
    assert sentence.tokens[2].end_position == 13

    sentence = Sentence(" I love  Berlin.", use_tokenizer=SegtokTokenizer())

    assert sentence.tokens[0].start_position == 1
    assert sentence.tokens[0].end_position == 2
    assert sentence.tokens[1].start_position == 3
    assert sentence.tokens[1].end_position == 7
    assert sentence.tokens[2].start_position == 9
    assert sentence.tokens[2].end_position == 15


def test_token_positions_when_creating_word_by_word():
    sentence: Sentence = Sentence(
        [
            Token("I"),
            Token("love"),
            Token("Berlin"),
            Token("."),
        ]
    )

    assert sentence.tokens[0].start_position == 0
    assert sentence.tokens[0].end_position == 1
    assert sentence.tokens[1].start_position == 2
    assert sentence.tokens[1].end_position == 6
    assert sentence.tokens[2].start_position == 7
    assert sentence.tokens[2].end_position == 13


@pytest.mark.skip(reason="New behavior no longer excludes line separators")
def test_line_separator_is_ignored():
    with_separator = "Untersuchungs-\u2028ausschüsse"
    without_separator = "Untersuchungs-ausschüsse"

    assert Sentence(with_separator).to_original_text() == Sentence(without_separator).to_original_text()


def no_op_tokenizer(text: str) -> list[str]:
    return [text]


def test_lazy_tokenization():
    # Test 1: Verify that sentences are not tokenized upon creation
    sentence = Sentence("The quick brown fox jumps over the lazy dog")
    assert sentence._tokens is None

    # Test 2: Verify that printing doesn't trigger tokenization on a sentence without token-labels
    str(sentence)  # Call str() to trigger printing
    assert sentence._tokens is None

    # Test 2b: Verify that adding token labels triggers tokenization
    sentence_with_token_label = Sentence("The quick brown fox jumps over the lazy dog")
    sentence_with_token_label[1].add_label("POS", "ADJECTIVE")
    assert sentence_with_token_label._tokens is not None

    # Test 2c: Verify that adding sentence labels does not trigger tokenization
    sentence_with_sent_label = Sentence("The quick brown fox jumps over the lazy dog")
    sentence_with_sent_label.add_label("POS", "VERB")
    assert sentence_with_sent_label._tokens is None

    # Test 3: Verify that iteration triggers tokenization
    sentence_iter = Sentence("The quick brown fox jumps over the lazy dog")
    assert sentence_iter._tokens is None
    for token in sentence_iter:
        pass
    assert sentence_iter._tokens is not None

    # Test 4: Verify that len() triggers tokenization
    sentence_len = Sentence("The quick brown fox jumps over the lazy dog")
    assert sentence_len._tokens is None
    _ = len(sentence_len)
    assert sentence_len._tokens is not None

    # Test 5: Verify that accessing tokens property triggers tokenization
    sentence_tokens = Sentence("The quick brown fox jumps over the lazy dog")
    assert sentence_tokens._tokens is None
    _ = sentence_tokens.tokens
    assert sentence_tokens._tokens is not None

    # Test 6: Verify that accessing text property does not trigger tokenization
    sentence_text = Sentence("The quick brown fox jumps over the lazy dog")
    assert sentence_text._tokens is None
    _ = sentence_text.text
    assert sentence_text._tokens is None


@pytest.mark.integration
def test_embeddings_tokenization():
    # Test 7: Verify that token-level embeddings triggers tokenization
    sentence_word = Sentence("The quick brown fox jumps over the lazy dog")
    word_embeddings = TransformerWordEmbeddings("distilbert-base-uncased")
    assert sentence_word._tokens is None
    word_embeddings.embed(sentence_word)
    assert sentence_word._tokens is not None

    # Test 8: Verify that sentence-level embeddings do not trigger tokenization
    sentence_doc = Sentence("The quick brown fox jumps over the lazy dog")
    doc_embeddings = TransformerDocumentEmbeddings("distilbert-base-uncased")
    assert sentence_doc._tokens is None
    doc_embeddings.embed(sentence_doc)
    assert sentence_doc._tokens is None


def test_remove_labels_keeps_untokenized():
    # Create a sentence without triggering tokenization
    sentence = Sentence("The quick brown fox jumps over the lazy dog")
    sentence.add_label("pos", "ADJ")
    assert not sentence._is_tokenized()  # Verify sentence starts untokenized

    # Remove labels should not trigger tokenization
    sentence.remove_labels("pos")
    assert not sentence._is_tokenized()  # Sentence should still be untokenized


def test_clear_embeddings_keeps_untokenized():
    # Create a sentence without triggering tokenization
    sentence = Sentence("The quick brown fox jumps over the lazy dog")
    assert not sentence._is_tokenized()  # Verify sentence starts untokenized

    # Clear embeddings should not trigger tokenization
    sentence.clear_embeddings()
    assert not sentence._is_tokenized()  # Sentence should still be untokenized


def test_create_sentence_with_staccato_tokenizer():
    sentence: Sentence = Sentence("I love Berlin.", use_tokenizer=StaccatoTokenizer())

    assert len(sentence.tokens) == 4
    assert sentence.tokens[0].text == "I"
    assert sentence.tokens[1].text == "love"
    assert sentence.tokens[2].text == "Berlin"
    assert sentence.tokens[3].text == "."


def test_staccato_tokenizer_with_numbers_and_punctuation():
    sentence = Sentence("It's 03-16-2025", use_tokenizer=StaccatoTokenizer())

    assert len(sentence.tokens) == 8
    assert [token.text for token in sentence.tokens] == ["It", "'", "s", "03", "-", "16", "-", "2025"]


def test_staccato_tokenizer_with_multilingual_text():
    # Test Russian
    russian_sentence = Sentence("Привет, мир! Это тест 123.", use_tokenizer=StaccatoTokenizer())
    assert [token.text for token in russian_sentence.tokens] == ["Привет", ",", "мир", "!", "Это", "тест", "123", "."]

    # Test Chinese
    chinese_sentence = Sentence("你好，世界！123", use_tokenizer=StaccatoTokenizer())
    assert [token.text for token in chinese_sentence.tokens] == ["你", "好", "，", "世", "界", "！", "123"]

    # Test Japanese
    japanese_sentence = Sentence("こんにちは世界！テスト123", use_tokenizer=StaccatoTokenizer())
    assert [token.text for token in japanese_sentence.tokens] == ["こんにちは", "世", "界", "！", "テスト", "123"]

    # Test Arabic
    arabic_sentence = Sentence("مرحبا بالعالم! 123", use_tokenizer=StaccatoTokenizer())
    assert [token.text for token in arabic_sentence.tokens] == ["مرحبا", "بالعالم", "!", "123"]


def test_sentence_retokenize():
    # Create a sentence with default tokenization
    sentence = Sentence("01-03-2025 New York")

    # Add span labels
    sentence.get_span(1, 3).add_label("ner", "LOC")
    sentence.get_span(0, 1).add_label("ner", "DATE")

    # Verify initial state
    assert len(sentence) == 3
    spans = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "01-03-2025"
    assert spans[1].text == "New York"

    # Retokenize with StaccatoTokenizer
    sentence.retokenize(StaccatoTokenizer())

    # Verify the sentence has more tokens after retokenization
    assert len(sentence) == 7

    # Verify the spans are preserved
    spans = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "01-03-2025"
    assert spans[1].text == "New York"

    # Verify the labels are preserved
    assert [label.value for label in spans[0].labels] == ["DATE"]
    assert [label.value for label in spans[1].labels] == ["LOC"]


def test_retokenize_with_complex_spans():
    # Test with more complex text and overlapping spans
    sentence = Sentence("John Smith-Johnson visited New York City on January 15th, 2023.")

    # Add span labels
    sentence.get_span(0, 2).add_label("ner", "PERSON")  # John Smith-Johnson
    sentence.get_span(3, 6).add_label("ner", "LOC")  # New York City
    sentence.get_span(7, 11).add_label("ner", "DATE")  # January 15th, 2023

    # Verify initial state
    assert len(sentence) == 12
    spans = sentence.get_spans("ner")
    assert len(spans) == 3
    assert spans[0].text == "John Smith-Johnson"
    assert spans[1].text == "New York City"
    assert spans[2].text == "January 15th, 2023"

    # Retokenize with StaccatoTokenizer
    sentence.retokenize(StaccatoTokenizer())
    assert len(sentence) == 15

    # Verify spans are preserved
    spans = sentence.get_spans("ner")
    assert len(spans) == 3
    assert spans[0].text == "John Smith-Johnson"
    assert spans[1].text == "New York City"
    assert spans[2].text == "January 15th, 2023"


def test_retokenize_preserves_sentence_labels():
    # Test that sentence-level labels are preserved
    sentence = Sentence("This is a positive review.")
    sentence.add_label("sentiment", "POSITIVE")

    # Verify initial state
    assert len(sentence.labels) == 1
    assert sentence.labels[0].value == "POSITIVE"

    # Retokenize
    sentence.retokenize(StaccatoTokenizer())

    # Verify sentence label is preserved
    assert len(sentence.labels) == 1
    assert sentence.labels[0].value == "POSITIVE"


def test_retokenize_multiple_times():
    # Test retokenizing multiple times
    sentence = Sentence("01-03-2025 New York")
    sentence.get_span(0, 1).add_label("ner", "DATE")
    sentence.get_span(1, 3).add_label("ner", "LOC")

    # First retokenization
    sentence.retokenize(StaccatoTokenizer())
    assert len(sentence) == 7

    # Second retokenization with a different tokenizer
    sentence.retokenize(SpaceTokenizer())
    assert len(sentence) == 3

    # Verify spans are still preserved
    spans = sentence.get_spans("ner")
    assert len(spans) == 2
    assert spans[0].text == "01-03-2025"
    assert spans[1].text == "New York"
