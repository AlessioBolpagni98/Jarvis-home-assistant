"""Test della spezzatura frasi in streaming (spec §13).

Verifica il contratto di ``SentenceSplitter``: emette una frase solo quando è
completa, tiene in buffer l'ultima (potenzialmente in costruzione), non spezza
decimali/abbreviazioni e forza il flush sulle frasi troppo lunghe. Niente modelli
né rete.
"""

from __future__ import annotations

from jarvis.sentences import SentenceSplitter


def _feed_all(splitter: SentenceSplitter, chunks: list[str]) -> list[str]:
    """Alimenta i chunk uno a uno e ritorna tutte le frasi emesse, flush incluso."""
    out: list[str] = []
    for c in chunks:
        out.extend(splitter.feed(c))
    out.extend(splitter.flush())
    return out


def test_emits_completed_sentence_keeps_last_buffered() -> None:
    s = SentenceSplitter(language="it")
    # La prima frase si chiude solo quando arriva testo della seconda.
    assert s.feed("Ciao.") == []  # "Ciao." da solo resta in buffer
    emitted = s.feed(" Come stai?")
    assert emitted == ["Ciao."]
    # "Come stai?" resta in buffer finché non si chiude lo stream.
    assert s.flush() == ["Come stai?"]


def test_full_text_reconstructed_char_by_char() -> None:
    s = SentenceSplitter(language="it")
    text = "Buongiorno. Oggi a Milano ci sono 18 gradi. Ti serve altro?"
    sentences = _feed_all(s, list(text))  # un carattere alla volta
    assert sentences == [
        "Buongiorno.",
        "Oggi a Milano ci sono 18 gradi.",
        "Ti serve altro?",
    ]
    assert " ".join(sentences) == text


def test_does_not_split_decimals() -> None:
    s = SentenceSplitter(language="it")
    sentences = _feed_all(s, ["Il caffè costa 3", ".", "14 euro. ", "Grazie!"])
    assert sentences == ["Il caffè costa 3.14 euro.", "Grazie!"]


def test_does_not_split_abbreviation() -> None:
    s = SentenceSplitter(language="it")
    sentences = _feed_all(s, ["Aspetta, ecc. ecc. e poi finiamo. Ok?"])
    assert sentences == ["Aspetta, ecc. ecc. e poi finiamo.", "Ok?"]


def test_max_chars_forces_flush_without_punctuation() -> None:
    s = SentenceSplitter(language="it", max_chars=20)
    # Nessun punto: la guardia spezza su uno spazio prima di max_chars.
    out = s.feed("una lunga sequenza di parole senza alcuna punteggiatura qui")
    assert out  # ha emesso almeno un pezzo senza aspettare un punto
    assert all(len(chunk) <= 20 for chunk in out)


def test_empty_feed_is_noop() -> None:
    s = SentenceSplitter(language="it")
    assert s.feed("") == []
    assert s.flush() == []
