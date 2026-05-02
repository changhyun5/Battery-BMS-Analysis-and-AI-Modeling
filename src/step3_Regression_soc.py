import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 불러오기
file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\5cycle_merged_data.csv'
df = pd.read_csv(file_path)

# 2.  SOC 12V=0%, 16.8V=100%
df['SOC_True'] = ((df['Stack Voltage'] - 12.0) / (16.8 - 12.0) * 100).clip(0, 100)

# 3. 데이터 분할 (1,2,3,4번 학습 / 5번 테스트)
train_df = df[df['Cycle'].isin([1, 2, 3, 4])]
test_df = df[df['Cycle'] == 5]

features = ['Cell 1 Voltage', 'Cell 2 Voltage', 'Cell 3 Voltage', 'Cell 4 Voltage', 'Current']
X_train, y_train = train_df[features], train_df['SOC_True']
X_test, y_test = test_df[features], test_df['SOC_True']

# 모델 학습 (Linear Regression)
model_soc = LinearRegression()
model_soc.fit(X_train, y_train)

# 예측 및 결과 출력
y_pred = model_soc.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"--- [SOC 회귀 모델 결과 (Test: Cycle 5)] ---")
print(f" 결정계수 (R2 Score): {r2:.4f}")
print(f" 평균 오차 (RMSE): {rmse:.2f}%")

# 시각화 및 저장
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual SOC', color='gray', alpha=0.5)
plt.plot(y_pred, label='AI Predicted', color='blue', linestyle='--')
plt.title('SOC Prediction (Train: Cycle 1-4 / Test: Cycle 5)')
plt.legend()
plt.savefig(r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\soc_result_cycle5.png')
plt.show()