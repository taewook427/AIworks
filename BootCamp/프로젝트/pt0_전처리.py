import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# raw 데이터 얻어오기
raw0 = pd.read_excel('./raw/demographics.xls', engine='xlrd')
print( "raw data info\n", raw0.info() )
raw0 = raw0[ ["ID", "Group", "Gender", "Age", "Height (meters)", "Weight (kg)", "UPDRS", "Speed_01 (m/sec)"] ]

# 전처리, 재작성
def readtxt(pid):
    with open(f"./raw/{pid}_01.txt", "r") as f:
        temp = [ x.split("\t") for x in f.readlines() ]
    chunk = [ ]
    for j in temp[0:12100]:
        dt = [0.0] * 16
        for k in range(0, 8):
            dt[k] = float(j[k+1])
            dt[k+8] = float(j[k+9])
        chunk.append(dt)
    return np.array(chunk)

temp = [ ]
for i in raw0.iloc:
    frag = [ ]
    frag.append( i["ID"] )
    if i["Group"] == "PD":
        frag.append(1)
    else:
        frag.append(0)
    if i["Gender"] == "male":
        frag.append(1)
    else:
        frag.append(0)
    frag.append( i["Age"] )
    if i["Height (meters)"] < 3:
        frag.append(i["Height (meters)"] * 100)
    else:
        frag.append( i["Height (meters)"] )
    frag.append( i["Weight (kg)"] )
    if np.isnan( i["UPDRS"] ):
        if "Co" in i["ID"]:
            frag.append(0.0)
        else:
            continue
    else:
        frag.append( i["UPDRS"] )
    frag.append( i["Speed_01 (m/sec)"] )
    if os.path.exists(f"./raw/{i['ID']}_01.txt"):
        frag.append( readtxt( i['ID'] ) )
    else:
        continue
    temp.append(frag)

raw1 = pd.DataFrame( {"id":[x[0] for x in temp],
                      "pd":[x[1] for x in temp],
                      "gender":[x[2] for x in temp],
                      "age":[x[3] for x in temp],
                      "height":[x[4] for x in temp],
                      "weight":[x[5] for x in temp],
                      "updrs":[x[6] for x in temp],
                      "speed":[x[7] for x in temp],
                      "force":[x[8] for x in temp]} )

# 체중 신장 속력 중앙값 결측치 채우기
print(raw1[raw1['pd'] == 0]["weight"].median())
print(raw1[raw1['pd'] == 1]["weight"].median())
print(raw1[raw1['pd'] == 0]["height"].median())
print(raw1[raw1['pd'] == 1]["height"].median())
print(raw1[raw1['pd'] == 0]["speed"].median())
print(raw1[raw1['pd'] == 1]["speed"].median())
raw1.loc[raw1['pd'] == 0, 'weight'] = raw1.loc[raw1['pd'] == 0, 'weight'].fillna(72.0)
raw1.loc[raw1['pd'] == 1, 'weight'] = raw1.loc[raw1['pd'] == 1, 'weight'].fillna(73.0)
raw1.loc[raw1['pd'] == 0, 'height'] = raw1.loc[raw1['pd'] == 0, 'height'].fillna(170.0)
raw1.loc[raw1['pd'] == 1, 'height'] = raw1.loc[raw1['pd'] == 1, 'height'].fillna(168.0)
raw1.loc[raw1['pd'] == 0, 'speed'] = raw1.loc[raw1['pd'] == 0, 'speed'].fillna(1.2485)
raw1.loc[raw1['pd'] == 1, 'speed'] = raw1.loc[raw1['pd'] == 1, 'speed'].fillna(1.075)

# IQR 1.5 이상치 제거
def outl_iqr(x, k=1.5):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (x > lower) & (x < upper), [lower, upper]

cond0, res0 = outl_iqr( raw1["age"] )
cond1, res1 = outl_iqr( raw1["height"] )
cond2, res2 = outl_iqr( raw1["weight"] )
cond3, res3 = outl_iqr( raw1["updrs"] )
cond4, res4 = outl_iqr( raw1["speed"] )
print("\nIQR 1.5 이상치 제거")
print(f"age : {res0}, height : {res1}, weight : {res2}, updrs : {res3}, speed : {res4}")
raw1 = raw1[cond0 & cond1 & cond2 & cond3 & cond4] ###
print( raw1.info() )

# 훈련, 테스트, 검증 데이터 나누기
from sklearn.model_selection import train_test_split

# train, validation, test (0.7, 0.2, 0.1)
bsinfo0, bsinfo1 = train_test_split(raw1, test_size=0.3, random_state=42, stratify=raw1["pd"])
bsinfo1, bsinfo2 = train_test_split(bsinfo1, test_size=0.33, random_state=42, stratify=bsinfo1["pd"])
print( bsinfo0.info(), bsinfo1.info(), bsinfo2.info() )

with open("./train.pkl", "wb") as f:
    pickle.dump(bsinfo0, f)
with open("./val.pkl", "wb") as f:
    pickle.dump(bsinfo1, f)
with open("./test.pkl", "wb") as f:
    pickle.dump(bsinfo2, f)
