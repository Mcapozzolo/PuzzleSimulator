# bsc-cs-pren1

## Set Up Environment

1. Clone this repository and navigate into it

2. [Install `uv`](https://github.com/astral-sh/uv)

3. Install dependencies:

   ```sh
   uv venv
   source .venv/bin/activate
   uv sync
   ```

4. Start the Raspberry Pi Controller with `uv run main.py [--development]`

5. (Install `ruff` to lint and format code. [VSCode Extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) or run `uvx ruff check` and `uvx ruff format`)

## `uv` - Package Manager

uv is a fast Python package manager that simplifies dependency management and virtual environments.

### Add Dependencies

```sh
uv add numpy
```
