import pandas as pd
import glob
from pathlib import Path


def merge_battery_cycles(data_folder, max_cycle=5):
    """
    지정된 폴더에서 충전/방전 데이터를 사이클(1~N) 순서대로 짝지어 병합하는 함수
    """
    df_list = []

    # 사이클부터 지정한 사이클까지 반복
    for i in range(1, max_cycle + 1):

        charge_files = glob.glob(rf"{data_folder}\charge_Data_{i}_*.csv")
        discharge_files = glob.glob(rf"{data_folder}\discharge_Data_{i}_*.csv")

        # 충전 데이터 불러오기
        if charge_files:
            df_c = pd.read_csv(charge_files[0])
            df_c['Cycle'] = i
            df_c['Mode'] = 'Charge'
            df_list.append(df_c)

        # 방전 데이터 불러오기
        if discharge_files:
            df_d = pd.read_csv(discharge_files[0])
            df_d['Cycle'] = i
            df_d['Mode'] = 'Discharge'
            df_list.append(df_d)

    # 모든 데이터를 하나로 합치기
    if not df_list:
        print("데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        return None

    final_df = pd.concat(df_list, ignore_index=True)

    print(f" 총 {final_df['Cycle'].nunique()}개의 사이클, {len(df_list)}개의 파일 병합 완료!")
    print(f" 전체 데이터 개수: {len(final_df)}행\n")

    return final_df


if __name__ == '__main__':

    folder_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data'

    # 함수 실행
    df_total = merge_battery_cycles(folder_path, max_cycle=5)

    if df_total is not None:
        print("=== 병합된 데이터 미리보기 (앞 5줄) ===")
        print(df_total[['TimeStamp', 'Stack Voltage', 'Current', 'Cycle', 'Mode']].head())
        print("...")
        print("=== 병합된 데이터 미리보기 (뒤 5줄) ===")
        print(df_total[['TimeStamp', 'Stack Voltage', 'Current', 'Cycle', 'Mode']].tail())

        # 새로운 CSV 파일로 저장
        save_path = folder_path + r'\5cycle_merged_data.csv'

        df_total.to_csv(save_path, index=False)
        print(f"\n--- Done ---:\n {save_path}")