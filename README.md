# numeraz

## Setup
1. Install dependencies:
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - `uv pip install -r pyproject.toml`
1. Create Azure subscription
1. Create Azure resources: `sh scripts/setup_aml.sh`
1. Create data asset: `uv run prepare_data.py`
