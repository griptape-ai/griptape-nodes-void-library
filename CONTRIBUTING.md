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

The library JSON's `pip_dependencies_exec` field is separate and **hand-maintained**: it lists what only `process()` needs, which for this library is the whole VOID inference stack. The engine installs it into `.venv-exec` beside the manifest, and the subprocesses this library launches use that interpreter. It has no `pyproject.toml` counterpart on purpose, because `uv` resolves every extra into one universal lock, so declaring the inference stack as an extra would drag its pins into the edit-time resolution. `make deps/sync` leaves the field untouched.

## CI

The CI workflow runs `make check` on every pull request and push to `main`. PRs must pass all checks before merging.
