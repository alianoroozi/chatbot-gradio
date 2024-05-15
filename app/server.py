import gradio as gr
from app.chat import ChatModelFactory


def predict(query, history, model_type='Claude', temperature=0.0):
    model = ChatModelFactory.create_model(model_type)
    yield from model.stream_response(query, history, temperature)


model_type_dropdown = gr.Dropdown(
    choices=[
        "Claude", "Openai", "Bedrock Llama 3", "Bedrock Claude", 
        "Bedrock Command R+", "Bedrock Mistral Large"
        ], 
    value="Claude", 
    label="Model Type"
)
temperature_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
gr.ChatInterface(
    predict,
    additional_inputs=[
        model_type_dropdown,
        temperature_slider
    ]
).launch(share=True)
