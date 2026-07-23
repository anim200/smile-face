#!/bin/sh
set -e

# Volumes mount empty on a new machine, so the model baked into the image is
# copied across on first boot. Later runs find a model already present and
# leave whatever the user has trained alone.
if [ ! -f /app/models/smile_clf.pkl ] && [ -f /app/seed/smile_clf.pkl ]; then
    echo "[entrypoint] Seeding model from image into empty volume"
    cp /app/seed/smile_clf.pkl /app/models/smile_clf.pkl
fi

# Sample photos, so the app can be demonstrated without hunting for faces.
if [ -d /app/seed/samples ] && [ ! -d /app/data/samples ]; then
    echo "[entrypoint] Seeding sample images"
    cp -r /app/seed/samples /app/data/samples
fi

mkdir -p /app/data/predictions /app/data/staging /app/data/dataset

echo "[entrypoint] Starting: $*"
exec "$@"