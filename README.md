# Chat bot with gradio interface

## Features:
- User can select between Openai (GPT4 Turbo) and Anthropic API (Claude Opus)
- Employs simple in-memory history to make the llm stateful
- Streams responses

## Environment variables
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

## How to run:
```
export PYTHONPATH="$PYTHONPATH:."
poetry run python app/app.py
```