# Variables
PROJECT_NAME = paper_bot
PYTHON_VERSION = python3.12
POETRY = poetry
PIP = pip
VENV_PATH = /root/.cache/pypoetry/virtualenvs/$(PROJECT_NAME)-jfmYsGBC-py3.12/bin/activate
CUDA_PATH = /usr/local/cuda
CUDA_BIN_PATH = $(CUDA_PATH)/bin
CUDA_LIB_PATH = $(CUDA_PATH)/lib64
PYTHON_INCLUDE_PATH = /usr/include/python3.12

# Default target
all: setup run

# Install system dependencies
deps:
    sudo apt update
    sudo apt install -y poppler-utils $(PYTHON_VERSION)-dev build-essential curl git

# Install Poetry
install-poetry:
    curl -sSL https://install.python-poetry.org | $(PYTHON_VERSION) -

# Initialize the project (clone repo, install deps)
setup: install-poetry deps
    git clone https://github.com/nikita-yatchenko/paper_bot.git || true
    cd $(PROJECT_NAME) && \
    $(POETRY) install --no-root && \
    $(POETRY) self add poetry-plugin-shell

# Activate virtual environment
activate:
    source $(VENV_PATH)

# Install flash-attn with no-build-isolation
install-flash-attn:
    $(PIP) uninstall -y flash-attn || true
    $(PIP) install flash-attn --no-build-isolation

# Set CUDA environment variables
set-cuda-env:
    export PATH=$(CUDA_BIN_PATH):$$PATH && \
    export LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$$LD_LIBRARY_PATH && \
    export CPATH=$$CPATH:$(PYTHON_INCLUDE_PATH)

# Run the bot
run: set-cuda-env
    cd $(PROJECT_NAME) && \
    source $(VENV_PATH) && \
    python bot.py

# Pull latest changes from Git
pull:
    cd $(PROJECT_NAME) && git pull

# Hard reset to origin/main
reset:
    cd $(PROJECT_NAME) && git reset --hard origin/main

# Clean up
clean:
    rm -rf $(PROJECT_NAME)

# List all commands
help:
    @echo "Available targets:"
    @echo "  all           - Setup and run the project"
    @echo "  deps          - Install system dependencies"
    @echo "  install-poetry - Install Poetry"
    @echo "  setup         - Clone repo and install dependencies"
    @echo "  activate      - Activate the virtual environment"
    @echo "  install-flash-attn - Install flash-attn"
    @echo "  set-cuda-env  - Set CUDA environment variables"
    @echo "  run           - Run the bot"
    @echo "  pull          - Pull latest changes from Git"
    @echo "  reset         - Reset to origin/main"
    @echo "  clean         - Clean up the project directory"
    @echo "  help          - Show this help message"

.PHONY: all deps install-poetry setup activate install-flash-attn set-cuda-env run pull reset clean help
