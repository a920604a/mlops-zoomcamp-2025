#! /bin/bash

python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r code/requirements.txt

# venv/bin/pytest tests/
# venv/bin/pylint --recursive=y .