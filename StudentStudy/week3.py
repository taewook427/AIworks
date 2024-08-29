import numpy as np
import matplotlib.pylab as plt

# 오차제곱합 손실함수
def sqr_err(res, ans):
    return 0.5 * np.sum( (res - ans) ** 2 )

# 교차 엔트로피 손실함수
def crs_ent(res, ans):
    return -np.sum( ans * np.log(res + 0.0000001) )

# 미니배치 교차 엔트로피
def crs_ent2(res, ans):
    if res.ndim == 1:
        res = res.reshape(1, res.size)
        ans = ans.reshape(1, ans.size)
    bsize = res.shape[0]
    return -np.sum( ans * np.log(res + 0.0000001) ) / bsize

# 수치 미분
def diff2(f, x):
    h = 0.0001
    return ( f(x + h) - f(x - h) ) / (2 * h)

# 다변수 그라디언트
def diff3(f, x):
    h = 0.0001
    grad = np.zeros_like(x)
    for i in range(0, x.size):
        tmp = x[i]
        x[i] = tmp - h
        y0 = f(x)
        x[i] = tmp + h
        y1 = f(x)
        grad[i] = (y1 - y0) / (2 * h)
        x[i] = tmp
    return grad

# 경사하강법
def descent(f, start_x, lr):
    x = start_x
    for i in range(0, 100):
        grad = diff3(f, x)
        x = x - grad * lr
    return x

def f0():
    t = np.array( [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] )

    y = np.array( [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] )
    print( sqr_err(y, t) )
    print( crs_ent(y, t) )

    y = np.array( [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0., 0.6, 0.0, 0.0] )
    print( sqr_err(y, t) )
    print( crs_ent(y, t) )

def f1():
    # y = x^3 + x^2, x = 5
    trif = lambda x: x ** 3 + x ** 2
    diff_trif = lambda x: 3 * x ** 2 + 2 * x
    print( diff2(trif, 5) - diff_trif(5) )

    dualf = lambda x: np.sum(x * x)
    print( diff3( dualf,  np.array( [3.0, 4.0] ) ) )
    print( diff3( dualf,  np.array( [3.0, -2.0] ) ) )
    print( diff3( dualf,  np.array( [-3.0, 8.0] ) ) )

    print( descent(dualf, np.array( [-3.0, 4.0] ), 0.1) ) # 적합 학습률
    print( descent(dualf, np.array( [-3.0, 4.0] ), 5) ) # 과대 학습률
    print( descent(dualf, np.array( [-3.0, 4.0] ), 0.0001) ) # 과소 학습률
