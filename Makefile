type:
	poetry run mypy .

run:
	poetry run python main.py

mic-check:
	poetry run python -m sounddevice