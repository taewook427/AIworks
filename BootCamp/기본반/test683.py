# 탐색적 데이터 분석

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# csv 불러오기
data = pd.read_csv('./healthcare-dataset-A.csv')
print( data.head(5) )
print( data.info() )
print( data.describe() )
data = data.rename(columns={'Residence_type' : 'residence_type'}) # 변수명수정

# 각 열 구성의 종류 (고유한 값만 출력)
for col_name in data.columns:
    if data[col_name].dtype in ['object', 'int64']:
        print(f"Unique values in column '{col_name}': {pd.unique(data[col_name])}")
data_float = data.select_dtypes( include=['float64'] ) # float64 형 데이터만 따로 뺀 데이터프레임
print( data_float.describe() )

data_new = data.copy() # 데이터 복제, 인코딩 매핑
data_new["gender_encoded"] = data["gender"].map( {'Male': 0, 'Female': 1, 'Other': 2} )
del data_new["gender"]
data_new["ever_married_encoded"] = data["ever_married"].map( {'No': 0, 'Yes': 1} )
del data_new["ever_married"]
data_new["work_type_encoded"] = data["work_type"].map( {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4} )
del data_new["work_type"]
data_new["residence_type_encoded"] = data["residence_type"].map( {'Urban': 0, 'Rural': 1} )
del data_new["residence_type"]
data_new["smoking_status_encoded"] = data["smoking_status"].map( {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3} )
del data_new["smoking_status"]
print( data_new.describe() )

# stroke 변수에 대한 히스토그램 시각화 -> 시각화 : GPT-generated
plt.figure(figsize=(10, 6))
plt.hist(data_new['stroke'], bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Stroke Variable')
plt.xlabel('Stroke')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 결측치 제거 데이터 생성
data_clean = data_new.dropna()
print( data_clean.describe() )
# 결측치 데이터 생성
data_null = data_new[ data_new['bmi'].isnull() ]
print( data_null.describe() )

# 나이대 기반 군집화
data_new['age_group'] = pd.cut(data_new['age'], bins=[0, 6, 12, 19, 39, 60, 100], labels=['0-6', '7-12', '13-19', '20-39', '40-60', '60+'], right=False)
print( data_new['age_group'].value_counts() )
# 파생변수 평균값, 중앙값
print( data_new.groupby("age_group", observed=True)["bmi"].mean(), data_new.groupby("age_group", observed=True)["bmi"].median() )

# 결측치 채우기 (40-60의 중앙값으로 채우기)
tofill = data_new.groupby("age_group", observed=True)["bmi"].mean()["40-60"]
data_filled = data_new.copy()
data_filled['bmi'] = data_filled['bmi'].fillna(tofill)
print( data_filled.describe() )

# IQR 이상치, Q1 - 1.5 * (Q3-Q1) ~ Q3 + 1.5 * (Q3-Q1)
def outl_iqr(x, k=1.5):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    # 이상치 여부를 나타내는 불리언 시리즈와 하한선 및 상한선 반환
    return (x < lower) | (x > upper), [lower, upper]
print( outl_iqr( data_filled["avg_glucose_level"] ) )
print( outl_iqr(data_filled["age"], k=0.5) )

# 이상치 제거 데이터
_, cond0 = outl_iqr( data_new["bmi"] )
_, cond1 = outl_iqr( data_new["avg_glucose_level"] )
data_filtered = data_new[ ( data_new["bmi"] > cond0[0] ) &
                          ( data_new["bmi"] < cond0[1] ) &
                          ( data_new["avg_glucose_level"] > cond1[0] ) &
                          ( data_new["avg_glucose_level"] < cond1[1] ) ]
print( data_filtered["bmi"].describe(), data_filtered["avg_glucose_level"].describe() )

# 상관계수
print( data_filtered[ ['age','avg_glucose_level', 'bmi'] ].corr() )

# csv로 저장 (healthcare-dataset-B.csv)
data_filtered.reset_index(drop=True, inplace=True)
data_filtered.to_csv('healthcare-dataset-B.csv', index=False)
