# 손글씨 인식 기본
import numpy as np

"""
훈련데이터 60000*784 행렬, 훈련라벨 60000*10 행렬
시험데이터 10000*784 행렬, 시험라벨 10000*10 행렬
이미지 데이터는 0~255 -> 0.0~1.0으로 정규화됨
"""
# mnist np.array train(60000, 784) test(10000, 784)
def getdata():
    train_img = [0] * 60000
    with open("train_img.bin", "rb") as f:
        data = f.read()
    for i in range(0, 60000):
        train_img[i] = list( data[784 * i:784 * i + 784] )
    train_img = np.array(train_img).astype(np.float32) / 255.0

    train_label = [0] * 60000
    with open("train_label.bin", "rb") as f:
        data = f.read()
    for i in range(0, 60000):
        train_label[i] = [0] * 10
        train_label[i][ data[i] ] = 1
    train_label = np.array(train_label)

    test_img = [0] * 10000
    with open("test_img.bin", "rb") as f:
        data = f.read()
    for i in range(0, 10000):
        test_img[i] = list( data[784 * i:784 * i + 784] )
    test_img = np.array(test_img).astype(np.float32) / 255.0

    test_label = [0] * 10000
    with open("test_label.bin", "rb") as f:
        data = f.read()
    for i in range(0, 10000):
        test_label[i] = [0] * 10
        test_label[i][ data[i] ] = 1
    test_label = np.array(test_label)
    
    return train_img, train_label, test_img, test_label

"""
순방향 x (N*M) 행렬 -> y (N*M) 행렬
역방향 d (N*M) 행렬 -> d (N*M) 행렬
"""
# ReLU 순전파/역전파 클래스
class relu():
    def __init__(self):
        self.mask = None # bool[][] np.array

    # x : np.2, out : np.2
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0 # 0 이상의 값만 내보냄
        return out

    # d : np.2, out : np.2
    def backward(self, d):
        d[self.mask] = 0 # 0 이상은 그대로, 나머지는 0을 반환
        return d

"""
순방향 x (N*M) 행렬 -> y (N*M) 행렬
역방향 d (N*M) 행렬 -> d (N*M) 행렬
"""
# sigmoid 순전파/역전파 클래스
class sigmoid():
    def __init__(self):
        self.out = None # float[][] np.array

    # x : np.2, out : np.2
    def forward(self, x):
        self.out = 1 / ( 1 + np.exp(-x) )
        return self.out

    # d : np.2, out : np.2
    def backward(self, d):
        return d * (1.0 - self.out) * self.out

"""
순방향 x (N*M), W (M*P), b (1*P) -> y (N*P)
역방향 d (N*P) -> db (1*P), dW (M*P), d (N*M)
"""
# 한 단계 전파 순전파/역전파 클래스
class affine():
    def __init__(self):
        self.x, self.W, self.b = None, None, None # y = x @ W + b
        self.dW, self.db = None, None

    # x : np.2, out : np.2
    def forward(self, x):
        self.x = x
        return (self.x @ self.W) + self.b

    # d : np.2, out : np.2
    def backward(self, d):
        self.db = np.sum(d, axis=0)
        self.dW = self.x.T @ d
        return d @ self.W.T

"""
순방향 x (N*M), t (N*M) -> y (N*M) -> loss (float)
역방향 d (1) -> d (N*M)
"""
# 소프트맥스/교차엔트로피 순전파/역전파 클래스
class soft():
    def __init__(self):
        self.loss = None # 손실함수 출력
        self.y = None # 소프트맥스 출력
        self.t = None # 원핫인코딩 정답라벨

    # x : np.2, t : np.2, out : float
    def forward(self, x, t):
        self.t = t

        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        self.y = y.T

        bsize = self.y.shape[0]
        self.loss = -np.sum( self.t * np.log(self.y + 0.0000001) ) / bsize
        return self.loss

    # d = 1, out : np.2
    def backward(self, d):
        bsize = self.t.shape[0]
        return (self.y - self.t) / bsize
