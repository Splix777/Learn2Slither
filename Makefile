# Variables
VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip
REQUIREMENTS := requirements.txt
LOGS := logs

.PHONY: all setup docker-setup start docker-start docker-clean docker stop clean

# Default target
all: docker

# Dockerized setup and start combined
docker: docker-setup docker-start

# Dockerized setup
docker-setup:
	docker compose build

# Start the application in Docker
# docker compose run learn2slither /bin/bash
docker-start:
	docker compose up -d
	docker compose exec learn2slither /bin/bash

# Clean up Docker (removes all images, volumes, networks, containers)
docker-clean:
	@echo "Pruning all Docker resources..."
	docker system prune -a --volumes -f

# Setup virtual environment and install requirements locally
setup: $(VENV_NAME)
	$(PIP) install -r $(REQUIREMENTS)

# Create virtual environment locally
$(VENV_NAME):
	python3 -m venv $(VENV_NAME)

# Start the application locally
start:
	$(PYTHON) cli.py

# Stop the application (placeholder for future use)
stop:
	@echo "Stop command not implemented."

# Clean up
clean:
	rm -rf $(VENV_NAME)
	rm -rf $(LOGS)