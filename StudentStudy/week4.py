import numpy as np

class add:
    def forward(self, x, y):
        return x + y

    def backward(self, d):
        return d, d

class mul:
    def __init__(self):
        self.x, self.y = None, None

    def forward(self, x, y):
        self.x, self.y = x, y
        return x * y

    def backward(self, d):
        return self.y * d, self.x * d

# 수치 미분
def num_gr(f, x):
    h = 0.0001
    grad = [0] * len(x)
    for i in range( 0, len(x) ):
        tmp = x[i]
        x[i] = tmp - h
        y0 = f(x)
        x[i] = tmp + h
        y1 = f(x)
        x[i] = tmp
        grad[i] = (y1 - y0) / (2 * h)
    return grad

# 사과 개수, 사과 가격, 귤 개수, 귤 가격, 소비세
def num_func(x):
    return ( x[0] * x[1] + x[2] * x[3] ) * x[4]

# 수치 미분으로 구한 d각요소/d최종가격 (최종가격이 1 움직일때 각 요소의 움직임)
def f0():
    x = [2, 100, 3, 150, 1.1]
    print(f"{num_func(x):.2f}")
    for i in num_gr(num_func, x): print(f"{i:.2f}", end=", ")

# 역전파로 구한 d각요소/d최종가격 (최종가격이 1 움직일때 각 요소의 움직임)
def f1():
    x = [2, 100, 3, 150, 1.1]
    c0, c1, c2, c3 = mul(), mul(), add(), mul()

    y = c3.forward( c2.forward( c0.forward( x[0], x[1] ), c1.forward( x[2], x[3] ) ), x[4] )
    print(f"{y:.2f}")

    grad = [0] * len(x)
    v0, grad[4] = c3.backward(1)
    v1, v2 = c2.backward(v0)
    grad[0], grad[1] = c0.backward(v1)
    grad[2], grad[3] = c1.backward(v2)
    for i in grad: print(f"{i:.2f}", end=", ")
