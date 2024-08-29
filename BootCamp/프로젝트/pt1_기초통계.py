import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("./train.pkl", "rb") as f:
    train = pickle.load(f)
with open("./val.pkl", "rb") as f:
    val = pickle.load(f)
with open("./test.pkl", "rb") as f:
    test = pickle.load(f)

# 데이터 형태, 힘 데이터 형태(n*16)
train.reset_index(drop=True, inplace=True)
val.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)
print("\n데이터 형태, 힘 데이터 형태(n*16)")
print( train.info(), val.info(), test.info() )
print( train["force"][0], train["force"][0].shape )

# 상관계수 히트맵
interested = ["gender", "age", "weight", "height", "updrs", "speed", "force"]
temp = pd.concat( [train, val, test], ignore_index=True)
temp['force'] = temp['force'].apply( lambda x: np.mean(x) )
sns.heatmap(temp[interested].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 데이터 최종 전처리 (정규화, 쪼개기)

# 나이, 신장, 속력 0~1 정규화
from sklearn.preprocessing import MinMaxScaler

tonorm0, tonorm1 = ["age", "height", "speed"], ["N_age", "N_height", "N_speed"]
scaler = MinMaxScaler()
train[tonorm1] = scaler.fit_transform( train[tonorm0] )
val[tonorm1] = scaler.fit_transform( val[tonorm0] )
test[tonorm1] = scaler.fit_transform( test[tonorm0] )
print("\n나이, 신장, 속력 0~1 정규화")
print( train.describe(), val.describe(), test.describe() )

# 0~5s 버리고 5s 단위 자르기 (gender N_age N_height weight speed N_speed updrs force)
def conv_data(data):
    temp = [ [ ], [ ], [ ], [ ], [ ], [ ], [ ], [ ] ]
    for i in data.iloc:
        frag = [ ]
        frag.append( i["gender"] )
        frag.append( i["N_age"] )
        frag.append( i["N_height"] )
        frag.append( i["weight"] )
        frag.append( i["speed"] )
        frag.append( i["N_speed"] )
        frag.append( i["updrs"] )

        arr = i["force"][500:]
        for j in range(0, len(arr) // 500):
            for k in range(0, 7):
                temp[k].append( frag[k] )
            temp[7].append( arr[500*j:500*j+500] )
    return pd.DataFrame( {"gender":temp[0], "N_age":temp[1], "N_height":temp[2], "weight":temp[3],
                          "speed":temp[4], "N_speed":temp[5], "updrs":temp[6], "force":temp[7]} )
train_new = conv_data(train)
val_new = conv_data(val)
test_new = conv_data(test)
print("\n데이터 형태, 힘 데이터 형태(500*16)")
print( train_new.info(), val_new.info(), test_new.info() )
print( train_new["force"][0], train_new["force"][0].shape )

# *_new를 이용할 것.
