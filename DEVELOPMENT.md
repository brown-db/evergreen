# Development

## Development Tools

Run Pyright:
```sh
uv run pyright
```

Run Ruff check:
```sh
uv run ruff check
```

Run Ruff format:
```sh
uv run ruff format
```

Run Scalene profiler:
```sh
uv run scalene run --profile-all --- -m pytest <path/to/test>
uv run scalene view
```

## Documentation Tools

This is primarily used to generate documentation for an LLM (see `docs/api_for_llms.md`).

Run MkDocs build:
```sh
uv run mkdocs build
```

Run MkDocs serve:
```sh
uv run mkdocs serve
```
