import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a**4

x = np.arange(0, 10)
y = 2 * x


def regarding_figure():
    # fig = plt.figure()
    # axes = fig.add_axes([0,0,1,1])
    # axes.plot(x,y)

    # Data
    # a = np.linspace(0, 10, 11)
    # b = a ** 4
    # 1. create figure
    # fig = plt.figure()
    # 2. define the size of figure.
    # axes = fig.add_axes([0,0,1,1])
    # 3. implement data
    # axes.plot(a,b)
    # display figure.
    # plt.show()

    # large axes and small axes

    fig = plt.figure()

    # large axes
    axes1 = fig.add_axes([0, 0, 1, 1])
    plt1 = axes1.plot(a, b)

    plt1.set_xlim(0, 8)
    axes1.set_ylim(0, 8000)
    axes1.set_xlabel("A")
    axes1.set_ylabel("B")
    axes1.set_title("Power of 4")
    # plt.xlim(0,8)
    # plt.ylim(0,8000)
    # plt.xlabel('A')
    # plt.ylabel('B')
    # plt.title('Power of 4')

    # small axes
    # axes2 = fig.add_axes([0.2, 0.5, 0.25, 0.25])
    # axes2.plot(x,y)

    plt.show()


def figure_parameter():
    fig = plt.figure(dpi=200, figsize=(2, 2))
    # dpi = resolution of chart
    # figsize = size of chart
    axes1 = fig.add_axes([0, 0, 1, 1])
    axes1.plot(a, b)
    plt.savefig("new_figure.png", bbox_inches="tight")
    # bbox_inches = padding of exported chart
    plt.show()


if __name__ == "__main__":
    # regarding_figure()
    figure_parameter()
