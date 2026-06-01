#!/bin/bash
cd /docker/mlinfra
git config --global --add safe.directory /docker/mlinfra
git fetch --all
git reset --hard origin/main
docker compose up -d --build --force-recreate --remove-orphans