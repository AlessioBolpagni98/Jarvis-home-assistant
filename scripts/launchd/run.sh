#!/usr/bin/env bash
# Wrapper avviato dal LaunchAgent: tiene la macchina sveglia (caffeinate) e avvia
# Jarvis in modalità wake word always-on. La GPU/Neural Engine restano attive per
# LLM/STT; il coperchio deve restare aperto (topologia 1, spec §12).
set -euo pipefail

# Cartella del progetto = due livelli sopra questo script.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

# Forza la modalità wake word indipendentemente da config.toml.
export JARVIS_RUNTIME__MODE="wake_word"

# caffeinate: -d niente sonno display, -i niente idle sleep, -m niente sonno disco,
# -s niente sonno su alimentazione, -u simula attività utente.
exec caffeinate -dimsu uv run jarvis
