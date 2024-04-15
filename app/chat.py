from openai import OpenAI
import anthropic
from abc import ABC, abstractmethod
from dotenv import load_dotenv


load_dotenv()


OPENAI_MODEL_NAME = "gpt-4-turbo-preview"
CLAUDE_MODEL_NAME = "claude-3-opus-20240229"

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()

class ChatModel(ABC):
    @abstractmethod
    def stream_response(self, messages, temperature):
        pass

class OpenAIModel(ChatModel):
    def stream_response(self, messages, temperature):
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            stream=True
        )

        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

class ClaudeModel(ChatModel):
    def stream_response(self, messages, temperature):
        response = anthropic_client.messages.stream(
            model=CLAUDE_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )

        partial_message = ""
        with response as stream:
            for text in stream.text_stream:
                partial_message += text
                yield partial_message

class ChatModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == 'openai':
            return OpenAIModel()
        elif model_type == 'claude':
            return ClaudeModel()
        else:
            raise ValueError("model_type must be either 'openai' or 'claude'")
