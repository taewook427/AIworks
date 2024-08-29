# 판다스 사용법

import numpy as np
import pandas as pd

data0 = pd.read_csv("healthcare-dataset-A.csv")
print(data0.shape)
data1 = pd.Series( [97, 98, 99, 100], index=["a", "b", "c", "d"] )
print( data1.iloc[0], data1["b"] )



# 기본적인 인공지능 학습
class basic:
    def __init__(self):
        self.data = np.array( [ [580, 700, 810, 840], [374, 385, 375, 401] ] ) # x축, y축
        self.w, self.b = 2, 1 # 가중치, 편향
        self.predict = lambda x: self.w * x + self.b # 일차 예측
        self.loss = lambda: np.mean( np.square( self.data[1] - self.predict( self.data[0] ) ) ) # 오차 제곱 평균 손실함수

k0 = basic()
