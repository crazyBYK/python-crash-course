import matplotlib.pyplot as plt
import numpy as np


def basics():
    x = np.arange(0, 10)
    y = 2 * x
    print(x)
    print(y)

    plt.plot(x, y)
    plt.title("String title")
    plt.xlabel("x axios")
    plt.ylabel("y axios")
    plt.xlim(0, 10)
    plt.ylim(0, 20)
    plt.savefig("first_plot.png")
    plt.show()


if __name__ == "__main__":
    basics()
