.PHONY: test check coverage htmlcov clean-coverage

test: check

check:
	python3 -m pytest tests/ -v

coverage:
	python3 -m coverage run -m pytest tests/
	python3 -m coverage report

htmlcov:
	python3 -m coverage run -m pytest tests/
	python3 -m coverage html
	@echo "Coverage report written to htmlcov/index.html"

clean-coverage:
	rm -rf .coverage htmlcov/
