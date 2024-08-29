# 손글씨 인식 기본
import numpy as np
import test671 as base

# 784 -(L1)-> 50 -(L2)-> 10
class net():
    def __init__(self, IsReLU):
        # 초기 파라미터로 학습률, 초기화계수, ReLU/sigmoid 여부 선택 가능
        self.epoch, self.initmul = 0.1, 0.01 # learning parm, random parm
        self.train_img, self.train_label, self.test_img, self.test_label = base.getdata()
        if IsReLU:
            self.layer1 = [ base.affine(), base.relu() ] # 784 -> 50
        else:
            self.layer1 = [ base.affine(), base.sigmoid() ] # 784 -> 50
        self.layer2 = [ base.affine(), base.soft() ] # 50 -> 10
        
        self.layer1[0].W = self.initmul * np.random.rand(784, 50)
        self.layer1[0].b = np.zeros(50, dtype=float)
        self.layer2[0].W = self.initmul * np.random.rand(50, 10)
        self.layer2[0].b = np.zeros(10, dtype=float)

    # forward with batch num
    def forward(self, num):
        temp = np.random.choice(60000, num)
        batch = np.zeros( (num, 784), dtype=float )
        ans = np.zeros( (num, 10), dtype=float )
        for i in range(0, num):
            batch[i] = self.train_img[ temp[i] ]
            ans[i] = self.train_label[ temp[i] ]

        # layer1 : N*784 -> N*50
        temp = self.layer1[0].forward(batch)
        temp = self.layer1[1].forward(temp)
        # layer3 : N*50 -> N*10 -> float
        temp = self.layer2[0].forward(temp)
        temp = self.layer2[1].forward(temp, ans)

        return temp # loss float

    # backward when doing learning
    def backward(self):
        # d (1) -> d (N*50)
        temp = self.layer2[1].backward(1)
        temp = self.layer2[0].backward(temp)
        # update layer2
        self.layer2[0].W = self.layer2[0].W - self.epoch * self.layer2[0].dW
        self.layer2[0].b = self.layer2[0].b - self.epoch * self.layer2[0].db
        # d (N*50) -> d (N*784)
        temp = self.layer1[1].backward(temp)
        temp = self.layer1[0].backward(temp)
        # update layer1
        self.layer1[0].W = self.layer1[0].W - self.epoch * self.layer1[0].dW
        self.layer1[0].b = self.layer1[0].b - self.epoch * self.layer1[0].db

    # learn with batch num
    def learn(self, num):
        # iter for 50 * 100
        for i in range(0, 50):
            self.forward(num)
            self.backward()

    # test with batch num
    def test(self, num):
        temp = np.random.choice(10000, num)
        batch = np.zeros( (num, 784), dtype=float )
        ans = np.zeros( (num, 10), dtype=float )
        for i in range(0, num):
            batch[i] = self.test_img[ temp[i] ]
            ans[i] = self.test_label[ temp[i] ]

        # layer1 : N*784 -> N*50
        temp = self.layer1[0].forward(batch)
        temp = self.layer1[1].forward(temp)
        # layer3 : N*50 -> N*10 -> float
        temp = self.layer2[0].forward(temp)
        temp = self.layer2[1].forward(temp, ans)

        # accuracy
        su = 0
        for i in range(0, num):
            if ans[i][ np.argmax( self.layer2[1].y[i] ) ] == 1:
                su = su + 1
        su = su / num

        return su, temp # loss float

"""
한번에 100개 배치로 정확도 테스트
1회 학습마다 100개 배치 * 50회 반복 (5000개 분량)
6만개 데이터 학습 100% 분량은 T12, 120% 분량은 T14~T15에서 달성됨
"""
k0, k1 = net(True), net(False)
for i in range(0, 50):
    print( f"T {i} ReLU : ", k0.test(100) )
    print( f"T {i} Sigmoid : ", k1.test(100) )
    k0.learn(100)
    k1.learn(100)
    print("")
