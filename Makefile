.PHONY: all test generate_embeddings format lint

generate_embeddings:
	mkdir -p data
	poetry run python scripts/generate_embeddings.py ../postman-docs/src/pages/docs

format:
	poetry run isort postman_docs_ai scripts
	poetry run black postman_docs_ai scripts

lint:
	poetry run ruff postman_docs_ai scripts

test:
	echo "No tests yet"

run-qdrant:
	docker run -p 6333:6333 qdrant/qdrant

run-api:
	poetry run uvicorn postman_docs_ai.search:app --reload --port 8001
