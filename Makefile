GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

format: ## run code formatter : black
		pre-commit install
		pre-commit run black --all-files
		# pre-commit run isort

# lint

test:   ## run tests with pytest
		pytest tests