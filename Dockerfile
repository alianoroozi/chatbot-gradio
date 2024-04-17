FROM python:3.11-slim

RUN pip install poetry

RUN poetry config virtualenvs.create false

WORKDIR /usr/src/app

COPY . .

RUN poetry install  --no-root

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV PYTHONPATH "${PYTHONPATH}:."

CMD ["python", "app/server.py"]