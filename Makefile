.PHONY: all test generate_embeddings format lint

generate_embeddings:
	poetry run python scripts/generate_embeddings.py ../postman-docs/src/pages/docs

format:
	poetry run isort postman_docs_ai scripts
	poetry run black postman_docs_ai scripts

lint:
	poetry run ruff postman_docs_ai scripts

test:
	echo "No tests yet"
