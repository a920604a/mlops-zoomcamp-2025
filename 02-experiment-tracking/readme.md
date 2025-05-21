mlflow ui --backend-store-uri sqlite:///mlflow.db

mlflow server --backend-store-uri sqlite:///mlflow.db  --host 127.0.0.1 --port 5000