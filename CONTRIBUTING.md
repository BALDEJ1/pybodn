# Contributing

Bug reports and ideas go to the [issues](https://github.com/BALDEJ1/pybodn/issues), where a
template asks for what helps. A vulnerability goes privately, as [SECURITY.md](SECURITY.md)
describes.

## Set up and check

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prek install    # secret and workflow checks before each commit
uv run ruff check
uv run ruff format --check
uv run mypy
uv run coverage run -m pytest
uv run coverage report
uv run python -m benchmarks.bench    # the figures under Speed and size, on this machine
```

The tests run with warnings as errors and must cover every line and branch. Every Python example
in the README runs as a test. CI runs the tests on Python 3.12 to 3.15, without NumPy, and on
SQLite 3.34, the oldest the README names.

## Branches

Branch from `develop` and open the pull request against it. `main` holds released versions only:
a release goes from a `release/X.Y.Z` branch into `main`, where it is tagged.

## A change

- Add a test that fails without the change.
- Keep the README and the docstrings true in the same pull request.
- Add a line under Unreleased in [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.
- Keep what the Compatibility section of the README covers. A change that cannot waits for 2.0.

## Writing

Write the README, docstrings, changelog, commit messages and pull requests in plain English
sentences that say what a thing does and when. A commit message starts with its type: `feat`,
`fix`, `docs`, `test`, `refactor`, `perf`, `ci` or `chore`.
