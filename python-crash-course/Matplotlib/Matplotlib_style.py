import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 11)


def styling():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(x, x, label="X vs X")
    ax.plot(x, x**2, label="X vs X^2")
    ax.legend(loc="0")
    # ax.legend(loc=(1.1, 0.5))
    # ax.legend(loc=(-0.1, 0.5))
    plt.show()


if __name__ == "__main__":
    styling()
