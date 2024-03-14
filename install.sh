#!/bin/bash

# Set environment variables and install the llama-cpp-python package with GPU support
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Install standard dependencies from requirements.txt
pip install -r requirements.txt
