# https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records

import pandas as pd
import numpy as np

# csv 로드, 필요한 값만 선택
# 나이, 빈혈, 크레아틴인산화효소, 당뇨, 심장방출혈액량, 고혈압, 혈소판, 성별, 크레아틴농도, 나트륨농도, 흡연, 추적시간, 사망
data = pd.read_csv("heart_failure_clinical_records.csv")
X = np.array( data.drop(columns=['time', 'DEATH_EVENT']) )
y = np.array(data['DEATH_EVENT']) # True 1

# 데이터 확인
print(data.mean())
print(data.mode())
print(data.max())
print(X.shape, X)
print(y.shape, y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization

# 딥러닝 모델 생성
model = Sequential()
model.add(Input(shape=(X.shape[1],)))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model.fit(X, y, epochs=16, batch_size=32, validation_split=0.2)

# 예측 코드, UI 상호작용
def ask():
    enc = np.array( [ [
        int(input("나이 : ")),
        1 if input("빈혈? (y/n) : ").lower() == "y" else 0,
        int(input("CPK (50~250) : ")),
        1 if input("당뇨? (y/n) : ").lower() == "y" else 0,
        int(input("심장방출혈액량 (52~74) : ")),
        1 if input("고혈압? (y/n) : ").lower() == "y" else 0,
        int(input("혈소판 (15~45) : "))*10000,
        float(input("크레아틴 (0.5~1.4) : ")),
        float(input("나트륨 (135~145) : ")),
        1 if input("남성? (y/n) : ").lower() == "y" else 0,
        1 if input("흡연? (y/n) : ").lower() == "y" else 0
    ] ] )
    dec = model.predict(enc, verbose=0)[0][0]
    print(f"심장마비 위험수치 : {dec:.4f}, 판단 : {dec>0.5}")

ask() # (80 y 450 y 30 n 20 4.5 130 y y), (70, n, 200, n, 60, n, 30, 1, 140, n, n)
