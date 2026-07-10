# Project metadata now lives in pyproject.toml (PEP 621).
# This shim remains so that legacy `python setup.py` / older-pip editable
# installs keep working; setuptools reads all fields from pyproject.toml.
from setuptools import setup

setup()
