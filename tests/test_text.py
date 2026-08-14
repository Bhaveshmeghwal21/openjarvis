from jarvis.text import approx_tokens, find_span, normalize


def test_normalize_collapses_whitespace():
    assert normalize("a   b\n\nc\t d") == "a b c d"


def test_normalize_repairs_hyphenation_across_line_breaks():
    assert normalize("distur-\nbance rejection") == "disturbance rejection"


def test_normalize_keeps_real_hyphens():
    assert normalize("state-of-the-art method") == "state-of-the-art method"


def test_normalize_folds_ligatures():
    assert normalize("ﬁne-tuned classiﬁer") == "fine-tuned classifier"


def test_normalize_folds_smart_quotes_and_dashes():
    assert normalize("“wind” — the agent’s") == '"wind" - the agent\'s'


def test_normalize_is_idempotent():
    once = normalize("distur-\nbance   “x”")
    assert normalize(once) == once


def test_find_span_locates_quote_despite_pdf_artifacts():
    haystack = "We show that distur-\nbance   rejection  improves."
    span = find_span("disturbance rejection", haystack)
    assert span is not None
    start, end = span
    assert normalize(haystack)[start:end] == "disturbance rejection"


def test_find_span_returns_none_for_absent_quote():
    assert find_span("never written", "some other text") is None


def test_find_span_is_case_and_punctuation_sensitive_enough_to_matter():
    # Fabrication must not pass: a different number is a different quote.
    assert find_span("94.2% on KITTI", "we report 91.7% on KITTI") is None


def test_find_span_rejects_a_partial_number_match_inside_a_longer_one():
    # "2.5% error" is a literal tail-substring of "12.5% error" but is a different,
    # fabricated number. An unanchored substring search would wrongly accept this.
    assert find_span("2.5% error", "we measure 12.5% error on the test set") is None


def test_find_span_rejects_a_partial_word_match():
    assert find_span("cat", "the results concatenate nicely") is None


def test_find_span_of_empty_needle_is_none():
    assert find_span("", "anything") is None


def test_approx_tokens_scales_with_length():
    assert approx_tokens("") == 0
    assert approx_tokens("one two three four") == 5  # 4 words * 1.3, rounded
    assert approx_tokens("a " * 100) > approx_tokens("a " * 50)
