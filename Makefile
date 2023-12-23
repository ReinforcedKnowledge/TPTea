GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# format
format:
		pre-commit install
		pre-commit run black --all-files
		pre-commit run isort

# lint

# test
test:
		pytest tests