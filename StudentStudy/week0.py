import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib

def f0():
    a = np.array( [1, 2, 3, 4] )
    print(a * 10, "\n")

    b = np.array( [ [1, 2], [3, 4] ] )
    print(b * 10, "\n")

    print(b * np.array( [10, 20] ), "\n")
    """
    1 2 * 10 20 -> 1 2 * 10 20
    3 4            3 4   10 20
    """

    print(b @ np.array( [ [10], [20] ] ), "\n")
    """
    1 2 @ 10 -> 1*10 + 2*20
    3 4   20    3*10 + 2*20
    """

    print(a > 2, "\n")

def f1():
    x = np.arange(0, 6, 0.1)
    y = np.sin(x)

    plt.plot(x, y)
    
    plt.show()

def f2():
    x = np.arange(0, 6, 0.1)
    y0 = np.sin(x)
    y1 = np.cos(x)

    plt.plot(x, y0, label="sin")
    plt.plot(x, y1, linestyle="--", label="cos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin & cos")
    plt.legend()
    
    plt.show()
