"""Segmentazione in frasi per lo streaming LLM→TTS (spec §8, FR8).

L'LLM emette testo a frammenti (``TextDelta``). Per far partire la voce sulla
**prima frase completa** mentre il modello continua a generare, serve rilevare i
confini di frase *in streaming*, su un buffer che cresce un pezzo alla volta.

``pysbd`` è un segmentatore batch (lavora su testo completo) con regole specifiche
per l'italiano: gestisce abbreviazioni (``ecc.``) e numeri decimali (``3.14``,
``15.30``) senza spezzarli. Lo incapsuliamo in uno **wrapper incrementale**:

- a ogni ``feed`` ri-segmentiamo l'intero buffer;
- emettiamo solo le frasi *seguite da altro testo* (``segments[:-1]``): l'ultimo
  segmento può essere ancora in costruzione, quindi resta nel buffer. Questo evita
  anche i falsi positivi sui decimali (``"Costa 3."`` da solo non viene emesso,
  perché non c'è nulla dopo: attende il resto e diventa ``"Costa 3.14 euro."``);
- una guardia ``max_chars`` forza il flush delle frasi lunghissime senza
  punteggiatura, così la voce non resta mai bloccata in attesa di un punto.
"""

from __future__ import annotations

import warnings

# pysbd 0.3.4 emette SyntaxWarning dai propri pattern regex: non è codice nostro.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd


class SentenceSplitter:
    """Accumula frammenti di testo ed emette frasi complete man mano.

    Uso::

        splitter = SentenceSplitter(language="it")
        for sentence in splitter.feed(delta_text):
            tts(sentence)
        for sentence in splitter.flush():  # coda finale
            tts(sentence)
    """

    def __init__(self, language: str = "it", max_chars: int = 200) -> None:
        # clean=False: pysbd non normalizza/scarta caratteri, il testo resta fedele.
        self._seg = pysbd.Segmenter(language=language, clean=False)
        self._max_chars = max_chars
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        """Aggiunge ``text`` al buffer e ritorna le frasi diventate complete."""
        if not text:
            return []
        self._buf += text
        out: list[str] = []

        segments = self._seg.segment(self._buf)
        if len(segments) > 1:
            out.extend(s.strip() for s in segments[:-1] if s.strip())
            self._buf = segments[-1]

        # Guardia anti-blocco: una frase troppo lunga senza confine viene spezzata
        # su uno spazio prima di max_chars (o di forza, se nemmeno quello c'è).
        while len(self._buf) > self._max_chars:
            cut = self._buf.rfind(" ", 0, self._max_chars)
            if cut <= 0:
                cut = self._max_chars
            chunk = self._buf[:cut].strip()
            if chunk:
                out.append(chunk)
            self._buf = self._buf[cut:].lstrip()

        return out

    def flush(self) -> list[str]:
        """Svuota il buffer a fine stream e ritorna l'eventuale frase residua."""
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []
