# Jarvis — assistente vocale locale

Assistente vocale **100% locale** in italiano: STT + LLM + TTS girano sulla macchina;
solo alcuni tool (meteo, ricerca) fanno rete. Vedi `assistente-vocale-locale-SPEC.md`
per il design completo.

Stadi distribuiti sulle tre unità di calcolo di Apple Silicon:
- **STT** Whisper large-v3-turbo → Neural Engine (Core ML)
- **LLM** Qwen3-30B-A3B (MoE) → GPU (via Ollama)
- **Wake word / VAD / TTS** → CPU

## Stato

- ✅ **Milestone 1** — MVP push-to-talk: INVIO → mic → Whisper → Qwen3 → Kokoro → audio.
- ✅ **Milestone 2** — streaming LLM→TTS: la voce parte sulla **prima frase** mentre l'LLM
  continua a generare. Segmentazione con `pysbd`, pipeline a 3 stadi su asyncio.
- ✅ **Milestone 3** — **tool calling**: registry estensibile (`@tool` + Pydantic) + loop
  agentico. Tre tool: `timer` (in-memory + notifica macOS), `meteo` (Open-Meteo, senza key),
  `web_search` (Brave). Aggiungere un tool = una funzione + `@registry.tool(...)`.
- ✅ **Milestone 4** — **wake word always-on**: ascolto continuo di «hey jarvis» (openWakeWord)
  + endpointing con Silero VAD. Selezionabile da config; servizio `launchd` opzionale.

## Prerequisiti

- **uv** (gestore progetto/Python) — già installato
- **Ollama** ≥ 0.19 — `brew install ollama`
- **Xcode Command Line Tools** (per il rebuild Core ML dell'STT) — `xcode-select --install`

## Setup

```bash
# 1. Dipendenze Python (crea .venv con Python 3.12)
uv sync

# 2. LLM: avvia Ollama e scarica un modello
ollama serve            # in un terminale a parte (o come servizio)
ollama pull qwen3.5:9b                    # leggero, per provare (~6.6 GB)
# ollama pull qwen3:30b-a3b-instruct-2507 # modello di produzione (~18 GB)

# 3. TTS: scarica i file di Kokoro nella root del progetto (~330 MB)
curl -L -o kokoro-v1.0.onnx  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 4. STT: il modello ggml di Whisper (~1.5 GB) si scarica da solo al primo avvio.
```

### (Opzionale ma consigliato) STT sul Neural Engine via Core ML

Il wheel di `pywhispercpp` da PyPI gira su Metal/GPU. Per spostare l'encoder di Whisper
sull'**ANE** (coerente col principio "GPU libera per l'LLM"), esegui lo script dedicato:

```bash
./scripts/setup_coreml.sh
```

Scarica l'encoder Core ML **già compilato** (niente Xcode completo), ricompila
pywhispercpp con `WHISPER_COREML=1` e corregge l'rpath del modulo. La **prima
trascrizione** è lenta: l'ANE compila il modello per il device; le successive sono rapide.

> Nota: `uv sync` mantiene il build da sorgente (stessa versione → "audited"), ma se
> ricrei `.venv` o aggiorni `pywhispercpp` rilancia lo script.

## Configurazione

Tutti i parametri sono in [`config.toml`](config.toml). I **segreti** vanno in variabili
d'ambiente con prefisso `JARVIS_` e separatore `__`:

```bash
export JARVIS_TOOLS__BRAVE_API_KEY=...   # per il tool web_search (Milestone 3)
```

### Tool (Milestone 3)

L'LLM decide autonomamente se usare un tool. Inclusi: `imposta_timer`/`elenca_timer`/
`cancella_timer` (locali, notifica macOS alla scadenza), `meteo` (Open-Meteo, nessuna
chiave: invia solo le coordinate), `web_search` (Brave, richiede la chiave sopra).
Nuovo tool = una funzione in [`src/jarvis/tools/`](src/jarvis/tools/) decorata con
`@registry.tool(...)` e aggiunta in `build_default_registry` — l'orchestratore non cambia.

### Modalità di interazione (Milestone 4)

`[runtime].mode` in `config.toml` sceglie come si parla con Jarvis:

- `push_to_talk` (default) — premi INVIO per parlare. Comodo per sviluppo/test.
- `wake_word` — ascolto continuo di «hey jarvis» (openWakeWord) con chiusura frase
  automatica via Silero VAD. Il modello `hey_jarvis` è già incluso in openWakeWord.

### Earcon di processing (latenza percepita)

Con il *thinking* dell'LLM abilitato (`think = true` in `[llm.extra_params]`) il tool
calling è più affidabile, ma cresce la latenza tra domanda e risposta. Per non lasciare
l'utente nel silenzio, durante l'elaborazione parte **in loop** un suono di tastiera che
si spegne nell'istante in cui inizia la voce del TTS: la latenza reale è identica, la
**percezione** è di un sistema reattivo. Configurazione in `[audio_feedback]`:

- `enabled` — attiva/disattiva l'earcon (default `true`).
- `processing_sound` — percorso del file (qualsiasi formato leggibile da `soundfile`).
- `volume` — guadagno applicato (es. `0.6` per renderlo più discreto rispetto alla voce).

Lo stream del suono è separato dalla coda di riproduzione del TTS (non si calpestano) ed
è tollerante ai guasti: se il file manca o non si decodifica, il turno prosegue senza
earcon. Implementazione in [`src/jarvis/audio_feedback.py`](src/jarvis/audio_feedback.py).

#### Always-on come servizio (`launchd`, opzionale)

Per far partire Jarvis al login e tenerlo sempre in ascolto (topologia 1: Mac sveglio,
coperchio aperto), c'è un LaunchAgent **già pronto ma non installato**:

```bash
./scripts/launchd/install.sh   # sostituisce i percorsi, copia il plist, carica il servizio
```

Lo script forza `mode = "wake_word"`, avvolge il processo in `caffeinate` (niente sonno)
e logga in `jarvis.{out,err}.log`. **Concedi prima il permesso microfono** eseguendo
`uv run jarvis` a mano una volta. Disinstallazione:
`launchctl bootout gui/$(id -u)/com.jarvis.assistant`.

## Avvio

```bash
uv run jarvis
```

Il comportamento dipende da `[runtime].mode` (vedi sopra):

- **push_to_talk** — premi **INVIO** per parlare, **INVIO** di nuovo per terminare.
- **wake_word** — di' «**hey jarvis**»: la cattura si chiude da sola al silenzio (VAD).

In entrambi i casi Jarvis trascrive, ragiona (chiamando i tool se serve) e risponde a
voce in streaming. `Ctrl-C` per uscire.

Per provare senza microfono (scrivi un prompt, ascolti la risposta in streaming):

```bash
uv run python scripts/smoke_streaming.py   # solo streaming LLM→TTS (M2)
uv run python scripts/smoke_agent.py       # loop agentico con tool (M3): timer/meteo/web
```

## Test

```bash
uv run pytest
```
