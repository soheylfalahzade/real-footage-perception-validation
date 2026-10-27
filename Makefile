.PHONY: install test lint typecheck all
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src tests
typecheck:
	mypy src/rfv
all: lint typecheck test
