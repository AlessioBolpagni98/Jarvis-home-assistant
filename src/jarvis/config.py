"""Configurazione tipizzata di Jarvis.

Carica `config.toml` e lo valida con pydantic-settings. I segreti si passano via
variabili d'ambiente con prefisso ``JARVIS_`` e separatore ``__`` per le sezioni
annidate (es. ``JARVIS_TOOLS__BRAVE_API_KEY``). L'env ha la precedenza sul TOML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Percorso del file di config, risolto rispetto alla root del progetto.
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


class AssistantConfig(BaseModel):
    system_prompt: str = "Sei Jarvis, un assistente vocale che parla italiano."


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 80
    input_device: str = ""
    output_device: str = ""

    @property
    def chunk_samples(self) -> int:
        """Numero di campioni per chunk (sample_rate * chunk_ms / 1000)."""
        return self.sample_rate * self.chunk_ms // 1000


class STTConfig(BaseModel):
    model: str = "large-v3-turbo"
    language: str = "it"
    n_threads: int = 4


class LLMConfig(BaseModel):
    # Il prefisso del modello seleziona il provider LiteLLM (es. "ollama_chat/...",
    # "openai/..."). Cambiare provider = cambiare questa riga (+ api_key via env).
    model: str = "ollama_chat/qwen3:30b-a3b-instruct-2507"
    api_base: str = "http://localhost:11434"  # "" per i provider cloud
    api_key: str = "ollama"  # placeholder per Ollama; per il cloud via env JARVIS_LLM__API_KEY
    temperature: float = 0.6
    max_tokens: int = 1024
    max_tool_iterations: int = 5
    request_timeout: float = 120.0
    drop_params: bool = True  # LiteLLM scarta i param non supportati dal provider attivo
    # Parametri provider-specific passati così come sono a LiteLLM (top-level). Es.
    # {"think": false} disattiva il ragionamento su Ollama; ignorato altrove (drop_params).
    extra_params: dict[str, Any] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    model_path: str = "kokoro-v1.0.onnx"
    voices_path: str = "voices-v1.0.bin"
    voice: str = "if_sara"
    lang: str = "it"
    speed: float = 1.0
    # Streaming: oltre questa lunghezza una frase senza punteggiatura viene
    # spezzata d'ufficio, così la voce non resta in attesa di un punto (spec §8).
    max_sentence_chars: int = 200


class AudioFeedbackConfig(BaseModel):
    # Earcon di processing: durante il thinking dell'LLM si riproduce in loop un
    # suono (tastiera) che si spegne all'inizio della voce — maschera la latenza.
    enabled: bool = True
    processing_sound: str = "sounds/dragon-studio-keyboard-typing-sound-effect-335503.mp3"
    volume: float = 1.0  # fattore di guadagno applicato al suono (0.0–1.0+)


class VADConfig(BaseModel):
    silence_ms: int = 500
    max_capture_s: int = 10


class WakeWordConfig(BaseModel):
    model: str = "hey_jarvis"
    threshold: float = 0.5


class ToolsConfig(BaseModel):
    weather_default_location: str = "Milano"  # meteo: località di default (spec §7.2)
    search_results: int = 5  # numero di risultati Brave passati all'LLM
    brave_api_key: str = ""  # solo via env JARVIS_TOOLS__BRAVE_API_KEY


class RuntimeConfig(BaseModel):
    # Modalità di interazione: push-to-talk (sviluppo) o wake word always-on (M4).
    mode: str = "push_to_talk"  # "push_to_talk" | "wake_word"


class LoggingConfig(BaseModel):
    level: str = "INFO"


class Config(BaseSettings):
    """Configurazione completa dell'applicazione."""

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_nested_delimiter="__",
        toml_file=CONFIG_PATH,
        extra="ignore",
    )

    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    audio_feedback: AudioFeedbackConfig = Field(default_factory=AudioFeedbackConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    wakeword: WakeWordConfig = Field(default_factory=WakeWordConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priorità: init > env > .env > TOML. L'env sovrascrive sempre il TOML.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_config() -> Config:
    """Carica e valida la configurazione."""
    return Config()
