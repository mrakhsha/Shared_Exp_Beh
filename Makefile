flake8:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 flake8
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

test:
	pytest --pyargs Shared_Exp_Beh --cov-report term-missing --cov=Shared_Exp_Beh