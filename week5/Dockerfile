FROM python:3.8.12-slim

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies directly to the system instead of create virtualization
RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]