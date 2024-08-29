# 주성분분석

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 결측치, 이상값 제거된 데이터
data = pd.read_csv('./healthcare-dataset-C.csv')
tgt0_raw = data[ ['age','avg_glucose_level', 'bmi'] ]
tgt1_raw = data["stroke"]

# 표준화 (평균이 0, 표준편차 1로 변경)
def std_fy(data):
    mean = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    z = (data - mean) / std
    return z
tgt0 = std_fy(tgt0_raw)
print( tgt0.describe() )

# PCA 모델 생성, 요소 2개로 압축 
pca = PCA(n_components=2)
X_pca = pca.fit_transform(tgt0)
print(X_pca)

# 주성분의 설명력
exp_ratio = pca.explained_variance_ratio_
print("주성분의 설명력 (분산 비율):")
for i, variance in enumerate(exp_ratio):
    print(f"PC{i+1}: {variance:.3f}")

# 주성분의 설명력을 시각화
plt.figure(figsize=(8,6))
plt.bar(range(1, len(exp_ratio) + 1), exp_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Each Principal Component')
plt.show()

### 시각화
new_PCA = pd.concat( [pd.DataFrame(X_pca), tgt1_raw], axis=1 ) # 종속변수(stroke)와 PCA화된 독립변수 결합
# stroke가 0/1인 항목 산점도
new_PCA_0 = new_PCA[new_PCA["stroke"] == 0]
new_PCA_1 = new_PCA[new_PCA["stroke"] == 1]
plt.scatter(new_PCA_0[0], new_PCA_0[1], c='blue', alpha=0.6, label='no_stroke')
plt.scatter(new_PCA_1[0], new_PCA_1[1], c='red', alpha=0.6, label='had_stroke')

plt.title('Stroke Map')  
plt.xlabel('PC1') # 주성분 1
plt.ylabel('PC2') # 주성분 2
plt.grid(True)
plt.legend()
plt.show()

# 새 데이터 포함 산점도
scaler = StandardScaler()
z_score = scaler.fit_transform(tgt0_raw.values)
new_data = [ [25, 80, 21] ] # 예측을 위한 새로운 데이터 (age, avg_glucose_level, bmi)
new_data = scaler.transform(new_data) # 새 데이터도 표준화 필요
new_data = pca.transform(new_data) # 표준화 후 PCA 변환

# 위와 동일 그래프 표시
plt.scatter(new_PCA_0[0], new_PCA_0[1], c='blue', alpha=0.6, label='no_stroke')
plt.scatter(new_PCA_1[0], new_PCA_1[1], c='red', alpha=0.6, label='had_stroke')
plt.scatter(new_data[0][0], new_data[0][1], c='black', s=70, alpha=1, label='State of stroke') # 새 데이터 표시

plt.title('Stroke Map')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.legend()
plt.show()

# 의사결정트리

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 독립종속변수 분리
X = data.drop(['id','stroke'], axis=1)
y = data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 및 학습 (랜덤시드 42, 최대깊이 5, 불순도계수 지니)
DT_clf_model = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='gini')
# 깊이, cpu수, 상태출력 설정
params = {'criterion': ['gini', 'entropy'],
          'max_depth': range(1, 21)}
grid_tree = GridSearchCV(
    DT_clf_model, 
    param_grid=params, 
    cv=5,
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
) # grid : 모든 경우의 수 테스트

# 학습, 결과 출력
grid_tree.fit(X_train, y_train)
y_pred = grid_tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
