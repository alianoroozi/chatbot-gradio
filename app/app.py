from openai import OpenAI
import anthropic
import gradio as gr
from dotenv import load_dotenv


load_dotenv()
OPENAI_MODEL_NAME = "gpt-4-turbo-preview"
CLAUDE_MODEL_NAME = "claude-3-opus-20240229"

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

def predict(query, history, model_type='openai'):
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human })
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": query})

    if model_type == 'openai':
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            temperature=0,
            stream=True
        )

        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message = partial_message + chunk.choices[0].delta.content
                yield partial_message

    elif model_type == 'claude':
        response = anthropic_client.messages.stream(
            model=CLAUDE_MODEL_NAME,
            max_tokens=1000,
            temperature=0,
            messages=[]
        )

        with response as stream:
            for text in stream.text_stream:
                yield text
    else:
        raise ValueError("model_type must be either 'openai' or 'claude'")


gr.ChatInterface(predict).launch(share=True)