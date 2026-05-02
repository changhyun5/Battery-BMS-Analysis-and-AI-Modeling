import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score


file_path = r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\5cycle_merged_data.csv'
df = pd.read_csv(file_path)


# 3.0V 조기 경고 정답
# 방전 중이고 셀 전압 중 하나라도 3.0V 이하로 떨어지면 위험(0), 아니면 정상(1)
def make_warning_label(group):
    labels = []
    for _, row in group.iterrows():
        min_v = row[['Cell 1 Voltage', 'Cell 2 Voltage', 'Cell 3 Voltage', 'Cell 4 Voltage']].min()

        # 방전 모드이면서 3.0V 이하일 때만 위험(0)으로 표시
        if row['Mode'] == 'Discharge' and min_v <= 3.0:
            labels.append(0)  # Warning
        else:
            labels.append(1)  # Normal
    return pd.Series(labels, index=group.index)


# 사이클별로 정답지 생성
df['Risk_True'] = df.groupby('Cycle').apply(make_warning_label).reset_index(level=0, drop=True)

# 3. 데이터 분할 (1,2,3,4번 학습 / 5번 테스트)
train_df = df[df['Cycle'].isin([1, 2, 3, 4])]
test_df = df[df['Cycle'] == 5]


features = ['Cell 1 Voltage', 'Cell 2 Voltage', 'Cell 3 Voltage', 'Cell 4 Voltage', 'Current']
X_train, y_train = train_df[features], train_df['Risk_True']
X_test, y_test = test_df[features], test_df['Risk_True']

# Logistic Regression

model_risk = LogisticRegression(max_iter=1000)
model_risk.fit(X_train, y_train)

# 예측 및 결과 출력
y_pred = model_risk.predict(X_test)

print(f"--- [3.0V 조기 경고 분류 결과 (Test: Cycle 5)] ---")
print(f" 모델 정확도 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f" F1-Score (종합 성적): {f1_score(y_test, y_pred):.4f}")
print(f"\n[혼동 행렬 (Confusion Matrix)]")
print(confusion_matrix(y_test, y_pred))

# 결과 해석
print("\n 상세 성능 리포트:")
print(classification_report(y_test, y_pred, target_names=['Warning(0)', 'Normal(1)']))

# 시각화
plt.figure(figsize=(12, 4))
plt.plot(y_test.values, label='Actual Label', color='gray', alpha=0.5, linewidth=5)
plt.plot(y_pred, label='AI Predicted', color='red', linestyle='--')
plt.title('3.0V Early Warning Prediction (Cycle 5)')
plt.yticks([0, 1], ['Warning(0)', 'Normal(1)'])
plt.legend()
plt.savefig(r'C:\Users\CENOTech\PycharmProjects\PythonProject9\data\warning_result_plot1.png')
plt.show()