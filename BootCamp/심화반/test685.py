# k-means clustering

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

# 임의데이터 k-means
def kmeans(X, K, max_iters=100):
    # 무작위 초기중심점
    centroids = X[ np.random.choice(a = range(X.shape[0]), size = K,  replace=False) ]
    print(f"클러스터 초기 중심 위치 : \n{centroids}")
    
    for i in range(max_iters):
        # 거리산출, 최소거리점 구하기
        distances = np.sum( (X[:, None] - centroids)**2, axis=-1 )
        labels = np.argmin(distances, axis = -1)

        # 새 중심점 구하기, 동일하다면 종료
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        print(f"\n클러스터링 업데이트 횟수 : {i+1}")
    return labels, centroids

# 데이터 읽고 kmeans 중심 3개 설정
df = pd.read_csv('./stroke_scaled.csv')
X = np.array(df)
labels, centroids = kmeans(X, 3)

# kmeans 결과 시각화
def visual0(X, labels, centroids):
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))  
    # 변수에 따른 인덱스 부여
    # 0 : age 
    # 1 : avg_glucose_level
    # 2 : bmi

    # 첫 번째 산점도: age & avg_glucose_level
    ax[0].scatter(X[:, 0], X[:, 1], alpha=0.6, c=labels, cmap='viridis')
    ax[0].scatter(centroids[:, 0], centroids[:, 1], s=150, c='red', label = 'centroids')
    ax[0].set_title('K-means Clustering: age & avg_glucose_level')
    ax[0].set_xlabel('age')
    ax[0].set_ylabel('avg_glucose_level')
    ax[0].grid()
    ax[0].legend()

    # 두 번째 산점도: age & bmi
    ax[1].scatter(X[:, 0], X[:, 2], alpha=0.6, c=labels, cmap='viridis')
    ax[1].scatter(centroids[:, 0], centroids[:, 2], s=150, c='red', label = 'centroids')
    ax[1].set_title('K-means Clustering: age & bmi')
    ax[1].set_xlabel('age')
    ax[1].set_ylabel('bmi')
    ax[1].grid()
    ax[1].legend()

    # 세 번째 산점도: avg_glucose_level & bmi
    ax[2].scatter(X[:, 1], X[:, 2], alpha=0.6, c=labels, cmap='viridis')
    ax[2].scatter(centroids[:, 1], centroids[:, 2], s=150, c='red', label = 'centroids')
    ax[2].set_title('K-means Clustering: avg_glucose_level & bmi')
    ax[2].set_xlabel('avg_glucose_level')
    ax[2].set_ylabel('bmi')
    ax[2].grid()
    ax[2].legend()

    plt.show()
visual0(X, labels, centroids) ###

# sklearn 사용 코드
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

# 3차원 산점도 시각화 함수
def visual1(X, labels, centroids):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 데이터 포인트 시각화
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', alpha=0.6)
    
    # 클러스터 중심 시각화
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=300, c='red', marker='X', label='centroids')
    
    ax.set_title('K-means Clustering 3D')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    plt.show()
visual1(X, kmeans.labels_, kmeans.cluster_centers_) ###

# 최적 k값 위한 elbow point
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    sse.append(kmeans.inertia_)

# Elbow Method 그래프 그리기
def visual2(sse):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sse)+1), sse, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid()
    plt.show()
visual2(sse) ###

# COPD 예측

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.utils import plot_model

# 데이터 불러오고 중복확인, 결측값 제거
data = pd.read_csv('./HealthProfileDataset_B_Train.csv')
print( data.info() )
print( len( pd.unique( data["ID"] ) ) )

# 방안 1. 'Height_cm' 열의 결측값 제거
data = data.dropna(subset=['Height_cm'])
# 방안 2. 'Height_cm'의 전체 평균값으로 대응
# data['Height_cm'] = data['Height_cm'].fillna(data['Height_cm'].mean())

# 방안 1. 'Weight_kg' 열의 결측값 제거
data = data.dropna(subset=['Weight_kg'])
# 방안 2. 'Weight_kg'의 전체 중앙값으로 대응
# data['Weight_kg'] = data['Weight_kg'].fillna(data['Weight_kg'].median())

# 방안 1. 'Cholesterol' 열의 결측값 제거
data = data.dropna(subset=['Cholesterol'])
# 방안 2. 'Cholesterol'결측값의 이전 행으로 대응
# data['Cholesterol'] = data['Cholesterol'].fillna(method='ffill')

# IQR 이상치 제거
def outliers_iqr(x, k=1.5):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return (x > lower_bound) & (x < upper_bound), [lower_bound, upper_bound]

columns_list = [
    # 'ID',
    'Age',
    'Height_cm',
    'Weight_kg',
    'BMI',
    'Vision',
    # 'Tooth decay',
    'fasting blood sugar',
    'Blood pressure',
    'Triglyceride',
    'Serum_Creatinine',
    'Cholesterol',
    'HDL_Cholesterol',
    'LDL_Cholesterol',
    'Hemoglobin',
    # 'Urine_Protein',
    'Liver_Enzyme',
    # 'COPD'
]
data_filtered = data.copy()
for i in range( 0, len(columns_list) ):
    outliers, bounds = outliers_iqr( data_filtered[ columns_list[i] ] )
    print(f"{columns_list[i]} : ({bounds[0]}, {bounds[1]})")
    data_filtered = data_filtered[outliers]
print( data_filtered.info() )
print( len( pd.unique( data_filtered["ID"] ) ) )

# 상관관계
data_corr = data_filtered[columns_list]
print( data_corr.corr() )
def visual3(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()
visual3(data_corr) ###

# 상관계수에 따른 변수제거, 독립종속변수 설정
clean_dataset = data_filtered.drop(['Height_cm', 'Weight_kg', 'Cholesterol'], axis=1)
X_dataset = clean_dataset.drop('COPD', axis=1)
X_train_dataset = X_dataset.drop('ID', axis=1)
Y_dataset = clean_dataset['COPD']
Y_train_dataset = Y_dataset

# 딥러닝 모델 생성
model = Sequential()
model.add(Input(shape=(X_train_dataset.shape[1],)))  # Input Layer
model.add(Dense(3, activation='relu'))    # Hidden Layers
model.add(Dense(1, activation='sigmoid')) # Output Layer

# 모델 컴파일, 훈련
t0 = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_dataset, Y_train_dataset, epochs=10, batch_size=32)
print(time.time() - t0)
print(model.layers[0].name)
print(model.layers[0].get_weights()[0])
print(model.layers[0].get_weights()[1])
print(model.layers[1].name)
print(model.layers[1].get_weights()[0])
print(model.layers[1].get_weights()[1])

# 정확도/손실함수 학습과정 시각화
def visual4(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)
    plt.show()

    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.grid(True)
    plt.show()
visual4(history) ###

# 새로운 데이터로 예측
new_data = pd.DataFrame({
    'Age': [50],
    'BMI': [25],
    'Vision': [1.0],
    'Tooth decay': [0],
    'fasting blood sugar': [100],
    'Blood pressure': [120],
    'Triglyceride': [150],
    'Serum_Creatinine': [1.0],
    'HDL_Cholesterol': [50],
    'LDL_Cholesterol': [130],
    'Hemoglobin': [15],
    'Urine_Protein': [0],
    'Liver_Enzyme': [0.5]
})
predictions = model.predict(new_data)
print(predictions)
for prediction in predictions:
    if prediction >= 0.5:
        print("COPD 있음 (확률:", prediction[0], ")")
    else:
        print("COPD 없음 (확률:", prediction[0], ")")
