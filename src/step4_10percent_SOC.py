import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, f1_score, recall_score
import warnings

warnings.filterwarnings('ignore')


# 데이터 불러오기 및 결측치 제거

file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\5cycle_merged_data.csv'
df = pd.read_csv(file_path).dropna()


# 통합 전처리 (10% 기준 정답지 생성)

# SOC 정답지 생성 (12.0V ~ 16.8V) 0~100%
df['SOC_True'] = ((df['Stack Voltage'] - 12.0) / (16.8 - 12.0) * 100).clip(0, 100)

# SOC가 10.0% 이하로 떨어지면 위험(0),아니면 정상(1)
df['Risk_True'] = df['SOC_True'].apply(lambda x: 0 if x <= 10.0 else 1)


# 데이터 분할 (Train: 1~4 / Test: 5)

train_df = df[df['Cycle'].isin([1, 2, 3, 4])]
test_df = df[df['Cycle'] == 5]

features = ['Cell 1 Voltage', 'Cell 2 Voltage', 'Cell 3 Voltage', 'Cell 4 Voltage', 'Current']

X_train_raw = train_df[features]
X_test_raw = test_df[features]

y_train_soc = train_df['SOC_True']
y_test_soc = test_df['SOC_True']

y_train_risk = train_df['Risk_True']
y_test_risk = test_df['Risk_True']


#  모델 학습

# 선형 회귀: 전압/전류를 보고 SOC(%)를 예측
model_soc = LinearRegression()
model_soc.fit(X_train_raw, y_train_soc)

y_train_pred_soc = model_soc.predict(X_train_raw)

# 로지스틱 회귀: 예측된 SOC(%)만 보고 위험(0/1) 여부를 학습
X_train_stacked = y_train_pred_soc.reshape(-1, 1)

model_risk = LogisticRegression(max_iter=1000, class_weight='balanced')
model_risk.fit(X_train_stacked, y_train_risk)


# 5. 모델 예측 및 평가 (Test: Cycle 5)

#  Test 데이터의 SOC 예측
y_pred_soc = model_soc.predict(X_test_raw)

# 예측된 Test SOC를 기반으로 위험 분류
X_test_stacked = y_pred_soc.reshape(-1, 1)
y_pred_risk = model_risk.predict(X_test_stacked)

# 지표 계산
rmse = np.sqrt(mean_squared_error(y_test_soc, y_pred_soc))
r2 = r2_score(y_test_soc, y_pred_soc)
acc = accuracy_score(y_test_risk, y_pred_risk)
recall_0 = recall_score(y_test_risk, y_pred_risk, pos_label=0)


print("\n" + "="*65)
print(" [SOC 10% 기준 지능형 방전 경고 모델 결과 (Cycle 5)]")
print("="*65)

print("\n [SOC 잔량 추정]")
print(f" - 평균 오차(RMSE): {rmse:.2f}%")

print("\n [잔량 10% 기준 경고 분류]")
print(f" - 예측 정확도    : {acc:.4f}")
print(f" - 위험 감지 재현율: {recall_0:.4f}")

threshold_soc = -model_risk.intercept_[0] / model_risk.coef_[0][0]
print(f"\n [모델 해석]: SOC가 '{threshold_soc:.2f}%' 이하부터 알람.")
print("="*65 + "\n")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# SOC 예측 그래프
ax1.plot(y_test_soc.values, label='Actual SOC', color='gray', alpha=0.6, linewidth=3)
ax1.plot(y_pred_soc, label='AI Predicted SOC', color='blue', linestyle='--', linewidth=1.5)
# 사용자가 설정한 10% 기준선
ax1.axhline(y=10.0, color='black', linestyle=':', label='Physical Limit (10%)')

ax1.axhline(y=threshold_soc, color='orange', linestyle='-', label=f'AI Trigger ({threshold_soc:.1f}%)')
ax1.set_title('[STEP 1] SOC Estimation & Warning Threshold', fontsize=14, fontweight='bold')
ax1.set_ylabel('State of Charge (%)', fontsize=12)
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)


ax2.plot(y_test_risk.values, label='Actual Risk (SOC <= 10%)', color='lightgray', linewidth=7, alpha=0.8)
ax2.plot(y_pred_risk, label='AI Predicted Risk', color='red', linestyle='--', linewidth=2)
ax2.set_title('[STEP 2] Energy-based (SOC <= 10%) Early Warning Classification', fontsize=14, fontweight='bold')
ax2.set_xlabel('Time Steps', fontsize=12)
ax2.set_ylabel('Status', fontsize=12)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Warning (Danger)', 'Normal'])
ax2.legend(loc='center left')
ax2.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
save_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\soc_10percent_warning_cycle5.png'
plt.savefig(save_path, dpi=300)
plt.show()