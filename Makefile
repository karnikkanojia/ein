.PHONY: lint format type-check test

lint:
	flake8 ein tests --max-line-length=88

format:
	black ein tests --line-length=88

type-check:
	mypy ein tests

test:
	pytest