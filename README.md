# Jarvis — Local Voice Assistant

> A fully **on-premise** voice assistant for Italian. STT, LLM, and TTS run entirely on your machine — no data ever leaves your hardware. One config line is enough to switch to any cloud LLM provider.

---

## Why Jarvis?

Most voice assistants depend on closed cloud APIs. Jarvis runs everything locally using open-source Python libraries:

| Stage | Library | Hardware (Apple Silicon) |
|---|---|---|
| **Wake word** | openWakeWord | CPU |
| **Speech → Text** | whisper.cpp (Core ML) | Neural Engine |
| **Reasoning + Tools** | LiteLLM + Ollama | GPU (MoE model) |
| **Text → Speech** | Kokoro ONNX | CPU |

All three Apple Silicon compute units work in parallel. If you prefer a cloud LLM (OpenAI, Anthropic, any OpenAI-compatible endpoint), it takes a single change in `config.toml` — the rest of the stack stays local.

---

## Features

- **100% on-premise by default** — Whisper, Qwen3, and Kokoro run locally; no account or API key required for the core
- **Streaming TTS** — voice playback starts on the first sentence while the LLM is still generating
- **Agentic tool calling** — LLM decides autonomously when to invoke tools; loop with configurable guard-rail
- **Extensible tool registry** — add a capability with one decorated function, no changes to the orchestrator
- **Wake word + VAD endpointing** — always-on "hey jarvis" detection, phrase boundary detected automatically by Silero VAD
- **Processing earcon** — typing sound masks LLM latency; stops the instant voice playback begins
- **Cloud LLM support** — drop-in swap via LiteLLM: `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`, any OpenAI-compatible endpoint
- **No API keys for tools** — weather via [Open-Meteo](https://open-meteo.com/), web search via DuckDuckGo (`ddgs`)

### Included tools

| Tool | What it does |
|---|---|
| `imposta_timer` / `elenca_timer` / `cancella_timer` | In-memory timers with macOS notification on expiry |
| `meteo` | Current weather for any city (Open-Meteo, no key) |
| `web_search` | Web search results (DuckDuckGo, no key) |

---

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) — Linux/Intel untested but mostly portable
- **Python 3.12+** managed by [uv](https://docs.astral.sh/uv/)
- **Ollama ≥ 0.19** for local LLM — `brew install ollama` (skip if using a cloud LLM)
- **Xcode Command Line Tools** for the Core ML STT rebuild — `xcode-select --install`

---

## Quick Start

```bash
# 1. Clone and install dependencies
git clone https://github.com/your-username/jarvis-home-assistant.git
cd jarvis-home-assistant
uv sync

# 2. Download the LLM (local mode)
ollama serve          # run in a separate terminal
ollama pull qwen3:4b  # lightweight, ~3 GB — good for testing

# 3. Download TTS model files (~330 MB, place in project root)
curl -L -o kokoro-v1.0.onnx  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 4. STT model (Whisper large-v3-turbo, ~1.5 GB) downloads automatically on first run.

# 5. Run
uv run jarvis
```

### (Recommended) Move STT to the Neural Engine

The default `pywhispercpp` wheel runs on Metal/GPU. To offload Whisper's encoder to the ANE and keep the GPU free for the LLM:

```bash
./scripts/setup_coreml.sh
```

The script downloads a pre-compiled Core ML encoder, recompiles `pywhispercpp` with `WHISPER_COREML=1`, and fixes the module rpath. The first transcription will be slow (one-time ANE compilation); subsequent runs are fast.

---

## Usage

Behavior depends on `[runtime].mode` in `config.toml`:

- **`push_to_talk`** *(default)* — press **Enter** to start recording, **Enter** again to stop.
- **`wake_word`** — say **"hey jarvis"**; the phrase closes automatically when silence is detected.

In both modes Jarvis transcribes your speech, reasons (calling tools if needed), and replies in streaming audio. Press `Ctrl-C` to exit.

```bash
# Test without a microphone (text prompt → streaming voice reply)
uv run python scripts/smoke_agent.py
```

### Always-on service (launchd)

To start Jarvis at login and keep it listening:

```bash
./scripts/launchd/install.sh
```

The script forces `wake_word` mode, wraps the process in `caffeinate`, and logs to `jarvis.{out,err}.log`. Grant microphone permission first by running `uv run jarvis` manually once.

---

## Configuration

All parameters live in [`config.toml`](config.toml). **Secrets go in environment variables**, never in the file:

```bash
export JARVIS_LLM__API_KEY=sk-...   # only needed for cloud LLM providers
```

### Switching LLM provider

Change the `[llm]` section — everything else stays the same:

**Local (Ollama, default)**
```toml
[llm]
model    = "ollama_chat/qwen3:30b-a3b-instruct-2507"
api_base = "http://localhost:11434"
api_key  = "ollama"
```

**Cloud (OpenAI)**
```toml
[llm]
model    = "openai/gpt-4o"
api_base = "https://api.openai.com/v1"
# api_key via JARVIS_LLM__API_KEY env var
```

**Any OpenAI-compatible endpoint**
```toml
[llm]
model    = "openai/your-model-name"
api_base = "https://your-endpoint/v1"
```

LiteLLM handles the translation. Provider-specific parameters (e.g. `think = true` for Ollama/Qwen3 reasoning mode) are declared per-provider and ignored automatically for providers that don't support them:

```toml
[llm.provider_extra_params.ollama_chat]
think = true   # enables reasoning on Qwen3; silently skipped for openai/*, etc.
```

### Key configuration sections

| Section | Purpose |
|---|---|
| `[assistant]` | System prompt — persona, language, style |
| `[stt]` | Whisper model variant, language, thread count |
| `[tts]` | Kokoro voice, speed, streaming chunk size |
| `[audio_feedback]` | Processing earcon — sound file, volume |
| `[vad]` | Silence threshold and max capture duration |
| `[wakeword]` | Wake word model and confidence threshold |
| `[tools]` | Default city for weather, search result count |

---

## Adding a Tool

```python
# src/jarvis/tools/my_tool.py
from .registry import ToolRegistry

def register(registry: ToolRegistry) -> None:
    @registry.tool(name="lights", description="Turn the lights on or off.")
    def lights(room: str, state: str) -> str:
        # your implementation
        return f"Lights in {room} are now {state}."
```

Then add `register(registry)` to `build_default_registry` in [src/jarvis/tools/\_\_init\_\_.py](src/jarvis/tools/__init__.py). The LLM will discover and call the tool automatically — no changes to the orchestrator.

---

## Project Structure

```
src/jarvis/
├── main.py            # entry point, interaction loop
├── agent.py           # agentic tool-calling loop
├── llm.py             # LiteLLM wrapper, streaming delta parser
├── audio_io.py        # microphone capture, audio playback queue
├── audio_feedback.py  # processing earcon
├── wake_word.py       # openWakeWord + Silero VAD
├── config.py          # pydantic-settings config model
└── tools/
    ├── registry.py    # @tool decorator and schema builder
    ├── timer.py       # timer tool
    ├── weather.py     # Open-Meteo weather tool
    └── web_search.py  # DuckDuckGo search tool
```

---

## Running Tests

```bash
uv run pytest
```

---

## License

MIT
