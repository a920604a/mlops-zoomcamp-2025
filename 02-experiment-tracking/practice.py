import mlflow
import pickle
import pandas as pd
import numpy as np
import logging
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import LinearSVR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("homework")


def read_dataframe(filename: str) -> pd.DataFrame:
    logging.info(f"Loading data from {filename}")
    df = pd.read_parquet(filename)
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    df["duration"] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    logging.info("Preparing features...")
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    logging.info(
        f"Features prepared: train shape {X_train.shape}, val shape {X_val.shape}"
    )

    return dv, X_train, X_val, y_train, y_val


def train_linear_model(X_train, y_train, X_val, y_val):
    logging.info("Training LinearRegression model...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    logging.info(f"LinearRegression RMSE: {rmse:.4f}")
    return lr, rmse


def train_lasso_model(X_train, y_train, X_val, y_val, alpha=0.1):
    logging.info(f"Training Lasso model with alpha={alpha}...")
    with mlflow.start_run():
        mlflow.set_tag("developer", "cristian")
        mlflow.log_param("alpha", alpha)

        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)

        y_pred = lasso.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        logging.info(f"Lasso RMSE: {rmse:.4f}")
    return lasso, rmse


def save_model(dv, model, filepath="models/lin_reg.bin"):
    logging.info(f"Saving model to {filepath}")
    with open(filepath, "wb") as f_out:
        pickle.dump((dv, model), f_out)


def objective(params, train, valid, y_val):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        y_pred = booster.predict(valid)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        logging.info(f"XGBoost trial with params {params} got RMSE: {rmse:.4f}")

    return {"loss": rmse, "status": STATUS_OK}


def tune_xgboost(X_train, y_train, X_val, y_val, max_evals=50):
    logging.info("Starting hyperparameter tuning for XGBoost...")

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:squarederror",
        "seed": 42,
    }

    trials = Trials()
    best_result = fmin(
        fn=lambda params: objective(params, train, valid, y_val),
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    logging.info(f"Best hyperparameters: {best_result}")
    return best_result


def train_other_models(X_train, y_train, X_val, y_val, dv):
    models = [
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
        LinearSVR,
    ]

    for model_class in models:
        with mlflow.start_run():
            model_name = model_class.__name__
            logging.info(f"Training {model_name}...")
            mlflow.set_tag("model", model_name)

            mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
            mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")
            # 假設 preprocessor 已經存在，若沒有此行可移除或改寫
            # mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

            model = model_class()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mlflow.log_metric("rmse", rmse)

            logging.info(f"{model_name} RMSE: {rmse:.4f}")


def main():
    logging.info("Start training pipeline")

    # 讀取資料
    df_train = read_dataframe(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
    )
    df_val = read_dataframe(
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"
    )
    logging.info(f"train rows: {df_train.shape[0]}, val rows: {df_val.shape[0]}")

    # 特徵工程
    dv, X_train, X_val, y_train, y_val = prepare_features(df_train, df_val)

    # 線性回歸模型訓練與儲存
    lr, rmse_lr = train_linear_model(X_train, y_train, X_val, y_val)
    save_model(dv, lr, "models/lin_reg.bin")

    # Lasso 訓練與記錄
    train_lasso_model(X_train, y_train, X_val, y_val, alpha=0.1)

    # XGBoost 超參數調優
    best_params = tune_xgboost(X_train, y_train, X_val, y_val, max_evals=50)

    # 其他模型訓練 (自動 logging)
    train_other_models(X_train, y_train, X_val, y_val, dv)

    logging.info("Training pipeline finished")


if __name__ == "__main__":
    main()
