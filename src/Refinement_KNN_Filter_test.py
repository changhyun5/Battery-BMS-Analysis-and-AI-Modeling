import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings('ignore')


# 1. 데이터 불러오기 및 필터링 (Cycle 4 방전 구간)

file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\step1_merged_data.csv'
df = pd.read_csv(file_path).dropna()

df_discharge = df[(df['Cycle'] == 4) & (df['Mode'] == 'Discharge')].copy()
df_discharge.reset_index(drop=True, inplace=True)


# 2. 방전 용량(Q) 적산 계산 (Ah 단위)

dt = 2.0  # 2초 간격
df_discharge['Real_Current'] = df_discharge['Current'].abs()
df_discharge['Capacity_Ah'] = (df_discharge['Real_Current'] * dt / 3600).cumsum()



# 3. KNN 노이즈 제거

def apply_knn_smoothing(data_series, n_neighbors):

    X_time = np.arange(len(data_series)).reshape(-1, 1)
    y_values = data_series.values

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_time, y_values)

    return knn.predict(X_time)



# 4. KNN 필터 적용 및 DVA 연산

# 전압 노이즈 제거
df_discharge['Smoothed_V_KNN'] = apply_knn_smoothing(df_discharge['Stack Voltage'], n_neighbors=31)

# 미분 연산 (dV/dQ)
df_discharge['dV'] = df_discharge['Smoothed_V_KNN'].diff()
df_discharge['dQ'] = df_discharge['Capacity_Ah'].diff()

# dQ가 0이거나 무한대 현상 방지
df_discharge = df_discharge[df_discharge['dQ'] > 0.0001].copy()
df_discharge['dV_dQ'] = df_discharge['dV'] / df_discharge['dQ']

# 미분값(dV/dQ) 노이즈 2차 제거 (이웃 데이터 51개 참조)
# 미분값은 전압보다 더 크게 튀므로, n_neighbors를 더 크게 주어 강하게 필터링
df_discharge['Smoothed_dV_dQ_KNN'] = apply_knn_smoothing(df_discharge['dV_dQ'], n_neighbors=51)


# 5. 방전 말기 변곡점 (Knee Point) 자동 탐색

# 방전 후반부(용량의 70% 이상 사용)에서 KNN으로 정제된 dV/dQ가 가장 가파르게 떨어지는 곳 찾기
threshold_capacity = df_discharge['Capacity_Ah'].max() * 0.70
late_discharge = df_discharge[df_discharge['Capacity_Ah'] > threshold_capacity]

knee_point = late_discharge.loc[late_discharge['Smoothed_dV_dQ_KNN'].idxmin()]
knee_capacity = knee_point['Capacity_Ah']
knee_voltage = knee_point['Stack Voltage']


# 6. 최종 결과 출력

print("\n" + "=" * 60)
print("  [KNN 필터링 기반 DVA 분석 결과 - Cycle 4]")
print("=" * 60)
print(f" - 해당 시점의 방전 용량 : {knee_capacity:.3f} Ah")
print(f" - 해당 시점의 실제 전압 : {knee_voltage:.2f} V")
print("=" * 60 + "\n")


# 7. 시각화 (KNN 정제 효과 및 DVA 피크)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# [Top] 전압 곡선 (Raw vs KNN Smoothed)
ax1.plot(df_discharge['Capacity_Ah'], df_discharge['Stack Voltage'], label='Raw Stack Voltage', color='lightgray',
         alpha=0.8, linewidth=2)
ax1.plot(df_discharge['Capacity_Ah'], df_discharge['Smoothed_V_KNN'], label='KNN Smoothed Voltage', color='blue',
         linewidth=2.5)
ax1.axvline(x=knee_capacity, color='red', linestyle='--', label=f'Knee Point ({knee_capacity:.2f} Ah)')
ax1.set_title('[STEP 1] Data Refinement: KNN Voltage Smoothing (Cycle 4)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Stack Voltage (V)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# [Bottom] 미분 전압 곡선 (Raw vs KNN Smoothed)
ax2.plot(df_discharge['Capacity_Ah'], df_discharge['dV_dQ'], label='Raw dV/dQ', color='lightgray', alpha=0.5)
ax2.plot(df_discharge['Capacity_Ah'], df_discharge['Smoothed_dV_dQ_KNN'], label='KNN Smoothed dV/dQ', color='purple',
         linewidth=2.5)
ax2.axvline(x=knee_capacity, color='red', linestyle='--', label='Physical 0% Indicator (Peak)')
ax2.set_title('[STEP 2] DVA Extraction using KNN-Smoothed Data', fontsize=14, fontweight='bold')
ax2.set_xlabel('Discharged Capacity Q (Ah)', fontsize=12)
ax2.set_ylabel('dV/dQ (V/Ah)', fontsize=12)
ax2.set_ylim(-15, 2)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\DVA_KNN_smoothed_cycle4.png'
plt.savefig(save_path, dpi=300)
plt.show()