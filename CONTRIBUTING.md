# Contributing

## Development Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install dev dependencies with:

```bash
make install/dev
```

To install all dependencies including core and extras:

```bash
make install
```

## Makefile Targets

Run `make` with no arguments to see all available targets.

### Checks

Run all checks (format, lint, types) before submitting a PR:

```bash
make check
```

Individual checks:

```bash
make check/format   # ruff format --check
make check/lint     # ruff check
make check/types    # pyright
make check/json     # validate JSON files with jq
```

### Fixing Issues

Auto-fix formatting and linting issues:

```bash
make fix
```

### Dependency Sync

The `pip_dependencies` field in the library JSON is kept in sync with `pyproject.toml`. Run this after adding or removing dependencies:

```bash
make deps/sync
```

This is also run automatically as part of `make install/core` and `make install/all`.

## CI

The CI workflow runs `make check` on every pull request and push to `main`. PRs must pass all checks before merging.
