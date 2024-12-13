#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.11.10
source .venv/bin/activate
uv pip install -r pyproject.toml
uv run pip install -e .
pre-commit install
