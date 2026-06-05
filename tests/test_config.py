"""Test della configurazione: caricamento TOML e override via variabili d'ambiente.

Non richiede modelli né rete: verifica solo il layer di config (pydantic-settings).
"""

from __future__ import annotations

from jarvis.config import Config, load_config


def test_loads_defaults_from_toml() -> None:
    cfg = load_config()
    # Valori che vivono in config.toml (verifica che il TOML venga letto)
    assert "/" in cfg.llm.model  # formato "provider/modello" di LiteLLM
    assert cfg.stt.language == "it"
    assert cfg.tts.lang == "it"
    # think vive in [llm.provider_extra_params.ollama_chat]: applicato solo per
    # modelli ollama_chat/*, ignorato per openai/* e altri provider.
    assert cfg.llm.provider_extra_params.get("ollama_chat", {}).get("think") is True


def test_chunk_samples_derived() -> None:
    cfg = load_config()
    # 16000 Hz * 80 ms / 1000 = 1280 campioni
    assert cfg.audio.chunk_samples == 1280


def test_env_overrides_toml(monkeypatch) -> None:
    monkeypatch.setenv("JARVIS_LLM__MODEL", "qwen3:4b")
    monkeypatch.setenv("JARVIS_TOOLS__SEARCH_REGION", "us-en")
    cfg = Config()
    assert cfg.llm.model == "qwen3:4b"
    assert cfg.tools.search_region == "us-en"
