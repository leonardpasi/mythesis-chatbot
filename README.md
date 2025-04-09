# mythesis-chatbot

This project uses **Poetry** for dependency management and environment setup.
Follow the instructions below to set up the environment and run the project.

## Installation

### Install Poetry

If you don't have **Poetry** installed, a nice way to do so is using [pipx](https://github.com/pypa/pipx).

```
pipx install poetry
```

### Install poetry plugins

1. poetry-plugin-dotenv: the plugin that automatically loads environment variables from a `.env` file into the environment before poetry commands are run
```bash
poetry self add poetry-plugin-dotenv
```

2. poetry-plugin-shell (optional): the plugin to run subshell with poetry virtual environment activated
```bash
poetry self add poetry-plugin-shell
```

3. Verify:
```bash
poetry self show plugins
```

### Create virtual environment

Once you've cloned the repo, from the project's root directory, install dependencies using Poetry:
```bash
poetry install
```
This will create a virtual environment and install all dependencies listed in the `poetry.lock` file.

### Configuring OpenAI API Key

1. Create a `.env` file in the root directory of the project:
```bash
touch .env
```

2. Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

**Note**: Make sure to keep your `.env` file private and not to commit it to version control.
It's included in `.gitignore` to prevent accidental commits.

### Running Scripts

To run scripts using Poetry:

```bash
poetry run python script_path/script_name.py
```

Optionally, create a subshell with Poetry’s virtual environment activated:
```bash
poetry shell
```
Then:
```bash
python script_path/script_name.py
```

### Verifying Setup

To verify that everything is set up correctly:

```bash
poetry run python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

This should load your API key from the `.env` file and print it.

### Install pre-commit hooks (for developers)
We use `black`, `isort` and `flake8` for formatting, import sorting, and linting. These
are run automatically at every commit through the installation of pre-commit hooks.
```bash
poetry run pre-commit install
```

