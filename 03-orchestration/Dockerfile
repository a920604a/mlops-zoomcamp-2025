FROM apache/airflow:2.8.2-python3.10

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

# 切換為 root 安裝系統相依套件
USER root

# 安裝 vim、wget 等工具
RUN apt-get update -qq && \
    apt-get install -yqq vim wget curl

# 安裝 OpenJDK 11
ENV JAVA_HOME=/home/jdk-11.0.2
ENV PATH="${JAVA_HOME}/bin/:${PATH}"

RUN DOWNLOAD_URL="https://download.java.net/java/GA/jdk11/9/GPL/openjdk-11.0.2_linux-x64_bin.tar.gz" \
    && TMP_DIR="$(mktemp -d)" \
    && curl -fL "${DOWNLOAD_URL}" --output "${TMP_DIR}/openjdk-11.0.2_linux-x64_bin.tar.gz" \
    && mkdir -p "${JAVA_HOME}" \
    && tar xzf "${TMP_DIR}/openjdk-11.0.2_linux-x64_bin.tar.gz" -C "${JAVA_HOME}" --strip-components=1 \
    && rm -rf "${TMP_DIR}" \
    && java --version

# 建立 Spark JARs 資料夾，並下載 BigQuery Connector
RUN mkdir -p /opt/spark/jars && chmod -R 777 /opt/spark/jars
RUN wget https://github.com/GoogleCloudDataproc/spark-bigquery-connector/releases/download/0.34.1/spark-3.4-bigquery-0.34.1.jar -O /opt/spark/jars/spark-bigquery-latest.jar

# 複製 Python 套件需求檔
COPY requirements.txt .

# 安裝 Python 套件（含指定版本 pendulum）
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install "pendulum==2.1.2" \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# 切回 airflow 預設使用者
USER $AIRFLOW_UID
