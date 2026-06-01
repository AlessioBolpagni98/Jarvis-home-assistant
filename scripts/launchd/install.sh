#!/usr/bin/env bash
# Installa il LaunchAgent di Jarvis (wake word always-on, spec §11 M4).
#
# NON viene eseguito automaticamente: lancialo tu quando vuoi rendere Jarvis un
# servizio che parte al login e resta in ascolto. Sostituisce i segnaposto nel
# template, copia il plist in ~/Library/LaunchAgents/ e lo carica con launchctl.
#
# Disinstallazione:
#   launchctl bootout gui/$(id -u)/com.jarvis.assistant
#   rm ~/Library/LaunchAgents/com.jarvis.assistant.plist
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LABEL="com.jarvis.assistant"
TEMPLATE="$PROJECT_DIR/scripts/launchd/$LABEL.plist.template"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

echo "Progetto:   $PROJECT_DIR"
echo "Destinazione: $DEST"

# Permessi microfono: un servizio launchd non può chiedere il consenso TCC in modo
# interattivo. Avvia almeno una volta 'uv run jarvis' a mano e concedi l'accesso al
# microfono al terminale, altrimenti la cattura fallisce in silenzio.
echo
echo "⚠️  Prima di procedere: esegui 'uv run jarvis' a mano almeno una volta e"
echo "    concedi l'accesso al microfono, così il permesso è già registrato."
read -r -p "Continuo con l'installazione? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Annullato."; exit 0; }

chmod +x "$PROJECT_DIR/scripts/launchd/run.sh"
mkdir -p "$HOME/Library/LaunchAgents"
sed "s#__PROJECT_DIR__#$PROJECT_DIR#g" "$TEMPLATE" > "$DEST"

# Ricarica pulita se già presente.
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL"

echo
echo "✅ Installato e avviato. Log: $PROJECT_DIR/jarvis.{out,err}.log"
echo "   Stato:   launchctl print gui/$(id -u)/$LABEL | head"
echo "   Stop:    launchctl bootout gui/$(id -u)/$LABEL"
