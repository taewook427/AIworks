import numpy as np

# 시그모이드 함수
def sigf(x):
    return 1 / ( 1 + np.exp(-x) )

# 항등함수
def idf(x):
    return x

# 소프트맥스 함수
def sof(x):
    x = x - np.max(x)
    temp = np.exp(x)
    return temp / np.sum(temp)
"""
Yi = exp(Ai) / Sum( exp(Ak) )
계산 전 각 수치에서 동일한 값을 빼도 결과가 같다.
모든 출력의 합은 1이다. -> 확률로 해석 가능
또한 지수함수는 단조증가라 대소관계가 바뀌지 않으므로, 분류문제에서는 생략 가능.
"""

# 2 -> 3 -> 2 -> 2, 3층 신경망
def parm_init_t():
    wdata = dict()
    wdata["W1"] = np.array( [ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ] )
    wdata["b1"] = np.array( [0.1, 0.2, 0.3] )
    wdata["W2"] = np.array( [ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6] ] )
    wdata["b2"] = np.array( [0.1, 0.2] )
    wdata["W3"] = np.array( [ [0.1, 0.3], [0.2, 0.4] ] )
    wdata["b3"] = np.array( [0.1, 0.2] )
    return wdata

# 순전파
def forward_t(wdata, x):
    a1 = x @ wdata["W1"] + wdata["b1"]
    z1 = sigf(a1)

    a2 = z1 @ wdata["W2"] + wdata["b2"]
    z2 = sigf(a2)

    a3 = z2 @ wdata["W3"] + wdata["b3"]
    y = idf(a3)

    return y

def f0():
    x = np.array( [1.0, 0.5] )
    print( forward_t(parm_init_t(), x) )
