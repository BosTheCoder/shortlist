# Show the available recipes.
default:
    @just --list

# Install/refresh the dev environment.
install:
    uv sync

# Lint, typecheck, test.
check: lint typecheck test

# Run the test suite.
test *ARGS:
    uv run pytest {{ARGS}}

# Lint and check formatting.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Static type check (pyright, basic mode).
typecheck:
    uv run pyright

# Format the code.
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Serve the web demo on http://localhost:8080.
demo:
    uv run uvicorn shortlist.web.app:app --host 127.0.0.1 --port 8080 --reload

# Regenerate the precomputed embedding vectors for the bundled datasets.
embed:
    uv run python -m shortlist.data.build_vectors

# Deploy the demo to Fly.
deploy:
    flyctl deploy
