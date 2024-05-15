from openai import OpenAI
import anthropic
import boto3
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from app.config import *


load_dotenv()


class ChatModel(ABC):
    @abstractmethod
    def stream_response(self, messages, temperature=0.0):
        pass

class OpenAIModel(ChatModel):
    def __init__(self):
        self.client = OpenAI()

    def stream_response(self, query, history, temperature=0.0):
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
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
    def __init__(self):
        self.client = anthropic.Anthropic()

    def stream_response(self, query, history, temperature=0.0):
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": query})

        response = self.client.messages.stream(
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

class BedrockModel(ChatModel):
    def __init__(self, model_id, params):
        self.client = boto3.client(service_name='bedrock-runtime')
        self.model_id = model_id
        self.params = params

    def stream_response(self, query, history, temperature=0.0):
        prompt = ""
        for human, assistant in history:
            prompt += f"Human: {human}\n\n Assistant:{assistant}\n\n"
        prompt += f"Human: {query}\n\n"

        body = {**self.params}
        body["prompt"] = prompt
        body["temperature"] = temperature

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id, 
            body=json.dumps(body)
        )

        stream = response.get('body')
        if stream:
            partial_message = ""
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    data = json.loads(chunk.get('bytes').decode())
                    text = data.get('generation', '')
                    partial_message += text
                    yield partial_message

class ClaudeBedrockModel(BedrockModel):
    def __init__(self, model_id, params):
        self.client = boto3.client(service_name='bedrock-runtime')
        self.model_id = model_id
        self.params = params

    def stream_response(self, query, history, temperature=0.0):
        messages = []
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": query})

        body = {**self.params}
        body["messages"] = messages
        body["temperature"] = temperature

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id, 
            body=json.dumps(body)
        )

        stream = response.get('body')
        if stream:
            partial_message = ""
            for event in stream:
                chunk = event.get('chunk')
                if chunk:
                    data = json.loads(chunk.get('bytes').decode())
                    text = data.get('generation', '')
                    partial_message += text
                    yield partial_message

class ChatModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == 'Openai':
            return OpenAIModel()
        elif model_type == 'Claude':
            return ClaudeModel()
        elif model_type == "Bedrock Llama 3":
            params = {
                "top_p": 0.9,
                "max_gen_len": 1024,
            }
            return BedrockModel(BEDROCK_LLAMA_MODEL_NAME, params)
        elif model_type == "Bedrock Claude":
            params = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
            }
            return ClaudeBedrockModel(BEDROCK_CLAUDE_MODEL_NAME, params)
#         elif model_type == "Bedrock Command R+":
#             {
#   "body": "{\"chat_history\":[{\"role\":\"USER\",\"message\":\"What is an interesting new role in AI if I don't have an ML background\"},
#   {\"role\":\"CHATBOT\",\"You could explore being a prompt engineer!\"}],
#   \"message\":\"What are some skills I should have\"}"
# }
#             return BedrockModel(BEDROCK_COMMAND_R_PLUS_MODEL_NAME)
        elif model_type == "Bedrock Mistral Large":
            params = {
                "max_tokens": 1024, 
                "top_p": 0.9, 
                "top_k": 50,
            }
            return BedrockModel(BEDROCK_MISTRAL_MODEL_NAME, params)
        else:
            raise ValueError("model_type must be either 'openai' or 'claude'")
