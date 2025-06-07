# Yellow Taxi ML Pipeline with Airflow & MLflow

## 專案介紹

本專案示範如何使用 Apache Airflow 建立一個機器學習資料處理與訓練的 Pipeline，結合 MLflow 進行模型追蹤與管理。  
主要流程包括從原始資料讀取、資料過濾、資料切分，最後進行線性回歸模型訓練並使用 MLflow 紀錄模型指標及參數。

專案同時透過 Docker Compose 管理整個服務，包含 Airflow、PostgreSQL、MLflow Server 與其資料庫。

---

## 專案架構

```
├── airflow/                    # Airflow DAGs 與設定
│   ├── dags/
│   │   └── yellow\_taxi\_pipeline.py   # Airflow DAG 定義
├── mlflow\_artifacts/           # MLflow 模型與 artifacts 儲存目錄 (volume)
├── postgres\_data/              # PostgreSQL Airflow 資料庫 volume
├── mlflow\_pg\_data/             # PostgreSQL MLflow 資料庫 volume
├── Dockerfile.airflow          # Airflow Dockerfile，安裝依賴
├── Dockerfile.mlflow           # MLflow Dockerfile，安裝依賴
├── docker-compose.yml          # Docker Compose 定義檔
├── requirements.txt            # Airflow 依賴套件
├── yellow\_taxi\_pipeline.py    # 主要 pipeline 與模型訓練邏輯
└── Makefile                   # 常用 Docker 服務管理指令

```

---

## 環境需求

- Docker (建議 20.10 以上)
- Docker Compose (版本 1.29 以上)
- GNU Make（可選，管理指令用）

---

## 快速啟動

1. Clone 本專案：

   ```bash
   git clone <你的專案網址>
   cd <專案資料夾>
    ```

2. 啟動服務：

   ```bash
   make up
   ```

3. 服務啟動後，訪問：

   * Airflow Web UI: [http://localhost:8080](http://localhost:8080)
     帳號密碼皆為 `airflow`

   * MLflow Tracking UI: [http://localhost:5001](http://localhost:5001)

---

## Docker 服務管理指令

| 指令             | 說明                                     |
| -------------- | -------------------------------------- |
| `make up`      | 啟動所有服務（背景執行）                           |
| `make down`    | 停止並刪除容器及資料卷                            |
| `make stop`    | 停止所有容器                                 |
| `make restart` | 重啟所有服務                                 |
| `make clean`   | 清除資料資料夾 (postgres, mlflow artifacts 等) |

---

## Airflow DAG 說明

* DAG 名稱：`yellow_taxi_pipeline`
* 無排程 (`schedule_interval=None`)，手動觸發
* 主要任務流程：

  1. 讀取原始 Yellow Taxi 資料（2023-03 月份）
  2. 資料過濾（Duration 1\~60 分鐘）
  3. 分割訓練/驗證資料
  4. 訓練線性迴歸模型並上傳 MLflow

---

## 模型訓練說明

* 使用 sklearn 的 LinearRegression
* 特徵為上車及下車地點（PULocationID、DOLocationID）
* 使用 MLflow 紀錄參數、模型與績效（RMSE、MAE、R2）
* 模型與 preprocessor (DictVectorizer) 會存成檔案並由 MLflow 追蹤管理

---

## 開發說明

* Airflow 依賴包列表請參考 `requirements.txt`
* MLflow 容器以官方映像為基底，額外安裝 psycopg2-binary 以支援 PostgreSQL
* 若需調整資料來源或模型邏輯，修改 `yellow_taxi_pipeline.py` 與 `airflow/dags/yellow_taxi_pipeline.py`

---

## 注意事項

* Airflow 使用 LocalExecutor，僅適合單機開發與測試
* PostgreSQL 資料庫與 MLflow artifacts 掛載至本地資料夾，避免資料遺失請勿隨意刪除資料夾
* MLflow Tracking URI 在 DAG 中預設為 `http://mlflow:5000`，容器間可透過此名稱連線

---
