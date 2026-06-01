"""Test della configurazione: caricamento TOML e override via variabili d'ambiente.

Non richiede modelli né rete: verifica solo il layer di config (pydantic-settings).
"""

from __future__ import annotations

from jarvis.config import Config, load_config


def test_loads_defaults_from_toml() -> None:
    cfg = load_config()
    # Valori che vivono in config.toml (verifica che il TOML venga letto)
    assert cfg.llm.model.startswith("ollama_chat/")  # prefisso provider LiteLLM
    assert "qwen3" in cfg.llm.model  # il tag esatto è tunabile
    assert cfg.stt.language == "it"
    assert cfg.tts.lang == "it"
    # think=false vive ora in [llm.extra_params] (knob provider-specific)
    assert cfg.llm.extra_params.get("think") is False


def test_chunk_samples_derived() -> None:
    cfg = load_config()
    # 16000 Hz * 80 ms / 1000 = 1280 campioni
    assert cfg.audio.chunk_samples == 1280


def test_env_overrides_toml(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_LLM__MODEL", "qwen3:4b")
    monkeypatch.setenv("JARVIS_TOOLS__BRAVE_API_KEY", "secret-123")
    cfg = Config()
    assert cfg.llm.model == "qwen3:4b"
    assert cfg.tools.brave_api_key == "secret-123"
