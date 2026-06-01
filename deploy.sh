#!/bin/bash
cd /docker/MLInfra
git config --global --add safe.directory /docker/MLInfra
git fetch --all
git reset --hard origin/main
docker compose up -d --build --force-recreate --remove-orphans