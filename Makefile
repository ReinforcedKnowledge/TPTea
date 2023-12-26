GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format: ## run code formatter : black
		pre-commit install
		pre-commit run black --all-files

lint:	## run linters: pre-commit (black, ruff) and mypy
		pre-commit install && pre-commit run --all-files --show-diff-on-failure
		mypy .

test:   ## run tests with pytest
		pytest tests