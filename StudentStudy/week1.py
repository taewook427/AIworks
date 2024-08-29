import numpy as np
import matplotlib.pylab as plt

def f0():
    # x0, x1을 받고 가중치 w0, w1을 곱해 lim 보다 크면 1을, 아니면 0을 내보내는 퍼셉트론
    pnode0 = lambda x0, w0, x1, w1, lim: 1 if x0 * w0 + x1 * w1 > lim else 0
    set0 = [ (0, 0), (0, 1), (1, 0), (1, 1) ]

    # and 게이트 퍼셉트론 예시
    for i in set0:
        print("AND", i, pnode0(i[0], 0.5, i[1], 0.5, 0.7) )
    print("")

    # x0, x1을 받고 가중치 w0, w1을 곱한 후 b를 더해 lim 보다 크면 1을, 아니면 0을 내보내는 퍼셉트론
    pnode1 = lambda x0, w0, x1, w1, b, lim: 1 if x0 * w0 + x1 * w1 + b > lim else 0

    # or 게이트 퍼셉트론 예시
    for i in set0:
        print("OR", i, pnode1(i[0], 0.3, i[1], 0.3, 0.1, 0.2) )
    print("")

    # xor 게이트 2층 퍼셉트론 예시
    for i in set0:
        # layer0 : OR, NAND
        x0, x1 = pnode0(i[0], 0.3, i[1], 0.3, 0.2), pnode0(i[0], -0.5, i[1], -0.5, -0.7)
        # layer1 : AND
        print("XOR", i, pnode0(x0, 0.5, x1, 0.5, 0.7) )
    print("")

"""
결국 퍼셉트론은 일차 관계를 묘사할 수 있습니다.
하지만 실제 인공지능이 다루는 문제는 매우 복잡합니다.
퍼셉트론을 여러 층으로 쌓아서 인간이 알고리즘으로 표현하기 힘든 문제도
그것을 해결하는 함수로 근사시킬 수 있다는 것이 딥러닝의 핵심입니다.
"""

def f1():
    # np.array를 받고 계단함수(a > 0) 결과물을 내보내는 함수
    stepf = lambda x: (x > 0).astype(np.int_)
    x = np.arange(-5.0, 5.0, 0.1)
    y = stepf(x)
    plt.plot(x, y)
    plt.ylim(-0.5, 1.5)
    plt.show()

def f2():
    # np.array를 받고 시그모이드 함수( 1/( 1+exp(-x) ) )를 내보냄
    sigf = lambda x: 1 / ( 1 + np.exp(-x) )
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigf(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

def f3():
    # np.array를 받고 ReLU 함수(0, x > 0)를 내보냄
    reluf = lambda x: np.maximum(x, 0)
    x = np.arange(-5.0, 5.0, 0.1)
    y = reluf(x)
    plt.plot(x, y)
    plt.ylim(-0.5, 5.5)
    plt.show()

def f4():
    A = np.array( [ [1, 2], [3, 4] ] )
    B = np.array( [ [5, 6], [7, 8] ] )
    print(A.shape)
    print( np.dot(A, B) )
    print(A @ B)

"""
신경망의 퍼셉트론은 다음과 같이 구성됩니다.
input에 가중치를 곱하고 편향을 더해서 a를 만드는 과정
+ 비선형 활성화 함수에 a를 넣은 값을 output으로 내보내는 과정

계단함수, 시그모이드, ReLU 등은 활성화 함수의 예시이고,
행렬곱을 통하여 한 레이어의 가중치 곱을 한번에 계산할 수 있습니다.
그 후 활성화 함수에 np.array를 넣어주면 순방향 한 단계 전파가 됩니다.

입력층 np.array -> 가중치 배열과 행렬곱 -> 활성화 함수 통과 -> 다음층 np.array
"""
