#!/bin/sh
set -e

DB=${DB_PATH:-/data/articles.db}
PARQUET=${PARQUET_PATH:-/data/sport_clanki.parquet}
EXISTING=${EXISTING_LABELS:-""}
READY="$DB.ready"

if [ ! -f "$READY" ]; then
    rm -f "$DB"
    echo "Ustvarjam bazo iz $PARQUET ..."
    if [ -n "$EXISTING" ] && [ -f "$EXISTING" ]; then
        python3 /app/prepare.py --input "$PARQUET" --existing-labels "$EXISTING" --db "$DB"
    else
        python3 /app/prepare.py --input "$PARQUET" --db "$DB"
    fi
    touch "$READY"
else
    echo "Baza ze obstaja: $DB"
fi

exec gunicorn \
    --bind "0.0.0.0:${PORT:-5050}" \
    --workers 1 \
    --timeout 120 \
    --access-logfile - \
    "app:app"
