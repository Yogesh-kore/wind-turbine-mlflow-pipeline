FROM python:3.10
RUN pip install mlflow
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlruns", "--host", "0.0.0.0", "--port", "5000"]