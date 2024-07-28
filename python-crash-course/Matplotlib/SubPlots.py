import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 10, 11)
b = a**4

x = np.arange(0, 10)
y = 2 * x


def create_subplots_ex():
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    #
    # axes[0][0].plot(x,y)
    # axes[0][1].plot(a,b)
    # plt.show()

    # fig, axes = plt.subplots(nrows=3, ncols=1)
    # for ax in axes:
    #     ax.plot(x,y)
    # plt.tight_layout()
    # plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(), dpi=200)

    axes[0][0].plot(x, y)
    axes[0][1].plot(x, y)
    axes[1][0].plot(x, y)
    axes[1][1].plot(a, b)

    plt.tight_layout()
    # fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axes[1][0].set_ylabel("Y LABEL 1,0")
    axes[1][1].set_title("TEST")

    fig.suptitle("Figure Level", fontsize=16)

    fig.savefig("new_subplots.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    create_subplots_ex()
