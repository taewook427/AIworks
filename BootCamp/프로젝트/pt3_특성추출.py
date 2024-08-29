# train_new, val_new, test_new 데이터 필요

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from re import L

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

def test(baseinfo_train):
    #foece data
    walkinfo_train = baseinfo_train['force']

    # 데이터 전처리
    rightfoot = np.array(walkinfo_train[:, 0:8]) # 8개의 열 force
    leftfoot = np.array(walkinfo_train[:, 8:]) # 8개의 열 force
    weight = baseinfo_train['weight'] # weight
    height = baseinfo_train['N_height'] # height -> 안쓰긴 합니다

    def find_first_zero(arr):
        # 발 뒤꿈치가 바닥에서 떨어지는 순간
        # 반환값: value가 index인 1차원 array
        """
        설명: 감지하는 값의 위치: allnotzero = True -> iscurzero = True 일 때.
        값이 바뀌는 순서: -> allnotzero = True -> iscurzero = False(초기화)-> iscurzero = True -> allnotzero = False -> (행 전체 검사 후) -> iscurzero = False(초기화) ->(iscurzero가 모두 False가 뜨면)
        """
        zero_pos = [ ]
        allnotzero = True
        for i in range(1, arr.shape[0]-1):
            iscurzero = False
            for j in range(8):
                if arr[i-1][j] == arr[i][j] == arr[i+1][j] == 0:
                    iscurzero = True
                    if allnotzero:
                        zero_pos.append(i)
                    allnotzero = False
                    
            if not iscurzero:
                allnotzero = True
        # 중복 제거 (같은 0을 여러 번 포함하지 않도록)
        zero_pos = sorted(set(zero_pos))
        return zero_pos

    def find_last_zero(arr):
      # 발 뒤꿈치가 바닥에 붙는 순간
      # 반환값: value가 index인 1차원 array

      zero_pos = [ ]
      allnotzero = True
      for i in range(1, arr.shape[0]-1):
        iscurzero = False
        for j in range(8):
          if arr[i-1][j] == arr[i][j] == arr[i+1][j] == 0:
            iscurzero = True
            allnotzero = False

        
        if not iscurzero:
            if not allnotzero:
                zero_pos.append(i)
                allnotzero = True

      # 중복 제거 (같은 0을 여러 번 포함하지 않도록)
      zero_pos = sorted(set(zero_pos))
      return zero_pos



    #featurs


    # feature 3. speed of walk; speed
    speed = baseinfo_train['speed']
    N_speed = baseinfo_train['N_speed']

    # feature 4. mean of stride; rmean, lmean
    rtimepoint1 = find_first_zero(rightfoot)
    ltimepoint1 = find_first_zero(leftfoot)

    rstride = []
    lstride = []

    for i in range(len(rtimepoint1)-1):
      rstride.append(rightfoot[rtimepoint1[i+1]] - rightfoot[rtimepoint1[i]])

    for i in range(len(ltimepoint1)-1):
      lstride.append(leftfoot[ltimepoint1[i+1]] - leftfoot[ltimepoint1[i]])


    rmean = np.mean(rstride)
    lmean = np.mean(lstride)


    #feature 5. variance of stride; rvar, lvar
    rvar = np.var(rstride)
    lvar = np.var(lstride)

    #feature 6. mean distance of stride; rdistance, ldistance

    rdistance = speed*rmean
    ldistance = speed*lmean

    # feature 7. difference of foot balance (상하); fh
    frontsum = np.sum(rightfoot[5:])
    heelsum = np.sum(leftfoot[0:3])

    diff = frontsum - heelsum
    fh = diff / weight


    # feature 8. ifference of foot balance (좌우); lr
    leftsum = np.sum(rightfoot[:, np.r_[2, 4, 6]])
    rightsum = np.sum(leftfoot[:, np.r_[1, 3, 5]])

    diff = leftsum - rightsum
    lr = diff / weight


    # feature 9 variance of force of steps; rstepvar, lstepvar

    rtimepoint2 = find_last_zero(rightfoot)
    ltimepoint2 = find_last_zero(leftfoot)

    rstepforce = []
    lstepforce = []

    # right foot sum
    for i in range(min(len(rtimepoint1), len(rtimepoint2))):
      if rtimepoint1[i] >= rtimepoint2[i]:
        start = rtimepoint2[i]
        end = rtimepoint1[i]
      else:
        start = rtimepoint1[i]
        end = rtimepoint2[i]

        force = rightfoot[start:end].sum()
        rstepforce.append(force)

    # left foot sum
    for i in range(min(len(ltimepoint1), len(ltimepoint2))):
      if ltimepoint1[i] >= ltimepoint2[i]:
        start = ltimepoint2[i]
        end = ltimepoint1[i]
      else:
        start = ltimepoint1[i]
        end = ltimepoint2[i]

        force = leftfoot[start:end].sum()
        lstepforce.append(force)

    # variance
    rstepvar = np.var(rstepforce)
    lstepvar = np.var(lstepforce)

    return rmean, rvar, rdistance, fh, lr, rstepvar

# train_new, val_new, test_new 데이터 필요

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
test_conv = extract(test_new)
