#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

cd /repo

pip install --upgrade pip
pip install --upgrade pipenv

pipenv install --dev
pipenv run black --target-version py38 --skip-string-normalization src
pipenv run python src/main.py
bash
