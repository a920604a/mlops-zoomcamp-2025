FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# 設定工作目錄
WORKDIR /app

# 複製 Pipfile 與 Pipfile.lock 並安裝依賴
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --system --deploy --ignore-pipfile

# 複製應用程式程式碼
COPY hw5.py .

# 預設執行指令
# CMD ["python", "hw5.py"]
