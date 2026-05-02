import pandas as pd
from pathlib import Path


def load_battery_data():
    # 파일 경로 설정
    log_path = Path(r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\discharge_Data_5_260413.log')

    # bqStudio 로그 읽기
    df = pd.read_csv(log_path, skiprows=2)

    rename_map = {
        'Cell 16 Voltage': 'Cell 4 Voltage',
        'CC2 Current': 'Current'
    }

    existing_cols = {old: new for old, new in rename_map.items() if old in df.columns}
    df.rename(columns=existing_cols, inplace=True)

    if existing_cols:
        print(f" 변경 완료: {list(existing_cols.keys())} -> {list(existing_cols.values())}")

    # 데이터 변환 (날짜, 전압, 전류)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # 단위 스케일링
    for col in df.columns:
        if 'Voltage' in col:
            if col == 'Stack Voltage':
                df[col] = df[col] / 100.0  # 10mV -> V
            else:
                df[col] = df[col] / 1000.0  # 1mV -> V
        elif col == 'Current':
            df[col] = df[col] / 1000.0  # 1mA -> A

    df = df.sort_values('TimeStamp').reset_index(drop=True)

    return df, log_path


if __name__ == '__main__':
    battery, original_path = load_battery_data()

    # CSV 저장
    output_path = original_path.parent / 'discharge_Data_5_260413.csv'
    battery.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f" CSV 저장 완료. 경로: {output_path}")