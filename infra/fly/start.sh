#!/usr/bin/env bash
# MANAGED BY demo-tools — DO NOT EDIT. Run `just sync` to update.
set -euo pipefail
APP="shortlist"
fly machines start --app "$APP"
echo "Started."
