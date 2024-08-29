# train_new, val_new, test_new 데이터 필요

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from re import L

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# 데이터 세트에서 추출
def fxx(data):
    return test(data)

# 데이터 세트 -> 특성치 데이터
def extract(data):
    # 특성 0 : 정규화 나이
    N_age = [ ]
    # 특성 1 : 원핫인코딩 성별
    gender = [ ]
    # 특성 2 : 정규화 스피드
    N_speed = [ ]
    # 특성 3 : 보폭시간의 평균 (정규화 필요)
    strmean = [ ]
    # 특성 4 : 보폭시간의 분산 (정규화 필요)
    strvar = [ ]
    # 특성 5 : 보폭거리의 평균 (정규화 필요)
    strdis = [ ]
    # 특성 6 : 상하 힘 차이 / 몸무게 (정규화 필요)
    updown = [ ]
    # 특성 7 : 좌우 힘차이 / 몸무게 (정규화 필요)
    leftright = [ ]
    # 특성 8 : (발자국 당 힘의 총합) / 몸무게의 분산 (정규화 필요)
    stepforce = [ ]

    v = [ ]

    for comp in data.iloc:
        N_age.append( comp["N_age"] )
        gender.append( comp["gender"] )
        N_speed.append( comp["N_speed"] )
        a, b, c, d, e, f = fxx(comp)
        strmean.append(a)
        strvar.append(b)
        strdis.append(c)
        updown.append(d)
        leftright.append(e)
        stepforce.append(f)
        v.append( comp["updrs"] )

    res = pd.DataFrame( {"N_age":N_age,
                   "gender":gender,
                   "N_speed":N_speed,
                   "N_strmean":strmean,
                   "N_strvar":strvar,
                   "N_strdis":strdis,
                   "N_updown":updown,
                   "N_leftright":leftright,
                   "N_stepforce":stepforce,
                   "updrs":v} )
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    tonorm = ["N_strmean", "N_strvar", "N_strdis", "N_updown", "N_leftright", "N_stepforce"]
    res[tonorm] = scaler.fit_transform( res[tonorm] )
    print( "\n딥러닝 전처리 완료 데이터 : ", res.info(), res.describe() )
    return res

train_conv = extract(train_new)
val_conv = extract(val_new)
test_conv = extract(test_new)print( "\n".join(temp0) )
minpos = np.argmin(temp1)
print( "minimal error at : ", temp0[minpos] )
plot_training_history( temp2[minpos] )
print( "각 인자별 (가중치, 편향) : ", temp3[minpos] )
