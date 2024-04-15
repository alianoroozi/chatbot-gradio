import gradio as gr
from app.chat import ChatModelFactory


def predict(query, history, model_type='claude', temperature=0.0):
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": query})

    model = ChatModelFactory.create_model(model_type)
    yield from model.stream_response(messages, temperature)


model_type_dropdown = gr.Dropdown(
    choices=["claude", "openai"], 
    value="claude", 
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
