#!/usr/bin/env bash
# MANAGED BY demo-tools — DO NOT EDIT. Run `just sync` to update.
set -euo pipefail
APP="shortlist"
DB_NAME="${APP}-db"
echo "==> Creating Fly Postgres cluster: $DB_NAME"
fly postgres create --name "$DB_NAME" --region lhr --vm-size shared-cpu-1x --volume-size 1 --initial-cluster-size 1
fly postgres attach --app "$APP" "$DB_NAME"
echo
echo "Postgres provisioned. DATABASE_URL has been added to the app's secrets."
