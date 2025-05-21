import mlflow

print("Q1. Install MLflow")

print(f"What's the version that you have? {mlflow.__version__}")


print("Q2. Download and preprocess the data")
import subprocess
import os


# python preprocess_data.py --raw_data_path <TAXI_DATA_FOLDER> --dest_path ./output
# How many files were saved to OUTPUT_FOLDER?
# Download dataset from

# 執行 bash 指令
cmd = "python preprocess_data.py --raw_data_path https://d37ci6vzurychx.cloudfront.net/trip-data/ --dest_path ./output"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# 印出執行結果（標準輸出）
print(result.stdout)

# 讀取 output 資料夾內檔案數量

output_folder = "./output"
if os.path.exists(output_folder):
    print(f"OUTPUT_FOLDER: {output_folder}")
else:
    os.makedirs(output_folder, exist_ok=True)
file_count = len(
    [
        f
        for f in os.listdir(output_folder)
        if os.path.isfile(os.path.join(output_folder, f))
    ]
)
print(f"How many files were saved to OUTPUT_FOLDER? {file_count}")


print("Q3. Train a model with autolog")
cmd = "python train.py --data_path ./output"

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)


print("Q4. Launch the tracking server locally")
cmd = "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 5000"

print("default-artifact-root")

print("Q5. Tune model hyperparameters")
cmd = "python hpo.py --data_path ./output --num_trials 15"

print("Q6. Promote the best model to the model registry")
cmd = "python register_model.py"
