# Chat bot with gradio interface

Todo: add gemini

## Features:
- User can select between Openai (GPT4 Turbo), Anthropic API (Claude Opus), and Llama 3
- Employs simple in-memory history to make the llm stateful
- Streams responses

## Environment variables
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

### For using Llama 3, you need to set up aws credentials and enable llama 3 model access in Amazon Bedrock.

## How to run:
```
export PYTHONPATH="$PYTHONPATH:."
poetry run python app/server.py
```
The app should now be accessible at http://localhost:7860.

## Build and Run Your Docker Container
```
docker build -t chatbot-gradio .
docker run -p 7860:7860 chatbot-gradio
```
The app should now be accessible at http://localhost:7860.