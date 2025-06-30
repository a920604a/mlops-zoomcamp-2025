import pandas as pd
from datetime import datetime
from deepdiff import DeepDiff
from batch import prepare_data

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),        # 9 分鐘 ✅
        (1, 1, dt(1, 2), dt(1, 10)),              # 8 分鐘 ✅
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),     # 59 秒 ❌ 小於 1 分鐘
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),         # 60 分 1 秒 ❌ 超過 60 分鐘
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    processed_df = prepare_data(df, categorical)

    # ✅ 預期只保留前兩筆，NaN 要填成 '-1'
    expected_data = [
        {'PULocationID': '-1', 'DOLocationID': '-1'},
        {'PULocationID': '1', 'DOLocationID': '1'},
    ]
    actual_data = processed_df[categorical].to_dict(orient='records')
    
    diff = DeepDiff(actual_data, expected_data, ignore_order=True)
    assert diff == {}, f"資料欄位不符，差異：{diff}"

    # ✅ 確認 duration 範圍
    assert processed_df['duration'].min() >= 1
    assert processed_df['duration'].max() <= 60

    # ✅ 確認筆數
    assert len(processed_df) == 2
