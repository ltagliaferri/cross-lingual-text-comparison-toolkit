"""
The spaCy- and Stanza-side extraction in the dependency scripts, driven by
small stub pipelines so no model (or spaCy/Stanza install) is needed.
"""

from types import SimpleNamespace

from cross_lingual_toolkit import cross_corpus_dependency as ccd
from cross_lingual_toolkit import dependency_parsing


# ---------------------------------------------------------------------------
# spaCy-like stubs
# ---------------------------------------------------------------------------

def _spacy_doc():
    """'God loves' — God (nsubj) -> loves (VERB)."""
    verb = SimpleNamespace(text="loves", lemma_="love", pos_="VERB", dep_="ROOT")
    verb.head = verb
    subj = SimpleNamespace(text="God", lemma_="God", pos_="PROPN", dep_="nsubj", head=verb)
    return [subj, verb]


def test_process_primary_matches_a_capitalised_configured_lemma():
    cfg = {"primary": {"agent_a_lemmas": ["God"], "agent_b_lemmas": ["Soul"]}}
    a_verbs, _, _, _ = ccd.process_primary(lambda text: _spacy_doc(), "God loves.", cfg, set())
    assert a_verbs == {"love": 1}


def test_extract_verbs_matches_a_capitalised_configured_lemma():
    counter = dependency_parsing.extract_verbs(_spacy_doc(), {"GOD"}, "nsubj")
    assert counter == {"love": 1}


def test_process_text_feeds_the_parser_whole_words():
    seen = []

    def nlp(piece):
        seen.append(piece)
        return []

    text = "alpha beta gamma delta. " * 50
    dependency_parsing.process_text(nlp, text, {}, chunk_size=100)
    assert "".join(seen) == text
    assert all(len(p) <= 100 for p in seen)
    for left, right in zip(seen, seen[1:]):
        assert left[-1].isspace() or right[0].isspace()


# ---------------------------------------------------------------------------
# Stanza-like stubs
# ---------------------------------------------------------------------------

def _stanza_nlp(deprel):
    """'Deus animam amat' with anima as `deprel` of amat (VERB)."""
    def nlp(text):
        words = [
            SimpleNamespace(id=1, text="Deus", lemma="deus", upos="NOUN", deprel="nsubj", head=3),
            SimpleNamespace(id=2, text="Animam", lemma="anima", upos="NOUN", deprel=deprel, head=3),
            SimpleNamespace(id=3, text="amat", lemma="amo", upos="VERB", deprel="root", head=0),
        ]
        return SimpleNamespace(sentences=[SimpleNamespace(words=words)])
    return nlp


CFG = {"comparison": {"agent_a_forms": ["Deus"], "agent_b_forms": ["animam"],
                      "name_forms": ["nomen"]}}


def test_process_comparison_counts_agent_b_as_subject_only():
    # regression: the comparison side counted objects as well, while the
    # primary side (and the CSV label) is subject-only
    _, b_verbs, _, _ = ccd.process_comparison(_stanza_nlp("obj"), "Deus animam amat.", CFG)
    assert b_verbs == {}


def test_process_comparison_counts_agent_b_as_subject():
    a_verbs, b_verbs, _, _ = ccd.process_comparison(
        _stanza_nlp("nsubj"), "Deus animam amat.", CFG)
    assert b_verbs == {"amo": 1}
    assert a_verbs == {"amo": 1}     # "Deus" configured capitalised still matches


# ---------------------------------------------------------------------------
# sentences_containing()
# ---------------------------------------------------------------------------

def test_sentences_containing_ignores_a_term_that_normalizes_to_nothing():
    # a soft-hyphen-only term would otherwise become an empty pattern
    # that matches every sentence
    text = "The abbey stands. The mill turns. Nothing here."
    assert list(ccd.sentences_containing(text, {"\u00ad", "mill"})) == ["The mill turns."]


def test_sentences_containing_keeps_sentences_that_share_an_opening():
    opening = "In the name of the most holy and undivided Trinity, " * 2   # > 100 chars
    s1 = opening + "we grant the abbey its lands."
    s2 = opening + "we confirm the abbey its mills."
    text = f"{s1} {s2} {s1}"
    found = list(ccd.sentences_containing(text, {"abbey"}))
    assert found == [s1, s2]


# ---------------------------------------------------------------------------
# Hardening review 2: NFC text and normalize_term()-keyed counters
# ---------------------------------------------------------------------------

import unicodedata  # noqa: E402

NFD_E = unicodedata.normalize("NFD", "é")


def test_process_text_feeds_the_parser_nfc_text():
    seen = []
    dependency_parsing.process_text(lambda p: seen.append(p) or [],
                                    f"caf{NFD_E} noir.", {}, chunk_size=100)
    assert seen == ["café noir."]


def test_extract_verbs_keys_counter_with_normalize_term():
    verb = SimpleNamespace(text="Créer", lemma_=f"Cr{NFD_E}er", pos_="VERB", dep_="ROOT")
    subj = SimpleNamespace(text="Dieu", lemma_="Dieu", pos_="PROPN", dep_="nsubj", head=verb)
    assert dependency_parsing.extract_verbs([subj, verb], {"dieu"}, "nsubj") == {"créer": 1}


def test_process_primary_parses_nfc_text_and_normalizes_keys():
    seen = []
    verb = SimpleNamespace(text="Créa", lemma_=f"Cr{NFD_E}er", pos_="VERB", dep_="ROOT")
    verb.head = verb
    subj = SimpleNamespace(text="Âme", lemma_=unicodedata.normalize("NFD", "âme"),
                           pos_="NOUN", dep_="nsubj", head=verb)
    adj = SimpleNamespace(text="Pure", lemma_="Pure", pos_="ADJ", dep_="amod", head=subj)

    def nlp(piece):
        seen.append(piece)
        return [adj, subj, verb]

    cfg = {"primary": {"agent_a_lemmas": ["x"], "agent_b_lemmas": ["âme"]}}
    _, b_verbs, _, b_adjs = ccd.process_primary(nlp, f"L'{NFD_E}me cr{NFD_E}a.", cfg, set())
    assert seen == ["L'éme créa."]
    assert b_verbs == {"créer": 1}
    assert b_adjs == {"pure": 1}


def test_process_comparison_matches_a_decomposed_name_form_in_the_text():
    nfd_name = unicodedata.normalize("NFD", "Hélène")
    seen = []

    def nlp(text):
        seen.append(text)
        words = [
            SimpleNamespace(id=1, text="Paris", lemma="Paris", upos="PROPN", deprel="nsubj", head=2),
            SimpleNamespace(id=2, text="Aima", lemma=f"aim{NFD_E}", upos="VERB", deprel="root", head=0),
            SimpleNamespace(id=3, text="Hélène", lemma="Hélène", upos="PROPN", deprel="obj", head=2),
        ]
        return SimpleNamespace(sentences=[SimpleNamespace(words=words)])

    cfg = {"comparison": {"agent_a_forms": ["x"], "agent_b_forms": ["y"],
                          "name_forms": ["hélène"]}}
    _, _, name_obj, _ = ccd.process_comparison(nlp, f"Paris aima {nfd_name}.", cfg)
    assert seen == ["Paris aima Hélène."]
    assert name_obj == {"aimé": 1}


def test_sentences_containing_matches_a_term_ending_in_a_vowel_sign():
    # "देवी" ends in a vowel sign (Mc), so \b after it never matched
    text = "यह देवी है. यह कुछ और है."
    assert list(ccd.sentences_containing(text, {"देवी"})) == ["यह देवी है."]


def test_sentences_containing_still_requires_whole_words():
    assert list(ccd.sentences_containing("Deusque. Deus amat.", {"deus"})) == ["Deus amat."]


def test_sentences_containing_ignores_an_empty_term():
    assert list(ccd.sentences_containing("One. Two.", {"", "two"})) == ["Two."]
