import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 11)


def styling2():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    # ax.plot(x,x, color='blue')
    # ax.plot(x,x, color='purple')

    # possible linestyle option '--', '-.', '-', ':', 'steps'
    ax.plot(
        x, x, color="#650bdb", label="x vs x", lw=3, ls="--", marker="+", ms=20
    )  # RGB HEX Code
    #
    ax.plot(
        x,
        x + 1,
        color="purple",
        label="x vs x+1",
        linewidth=3,
        linestyle="-.",
        marker="o",
        markersize=10,
        markerfacecolor="red",
        markeredgecolor="orange",
        markeredgewidth=1,
    )

    # lines = ax.plot(x, x+1, color='purple', lw=5)
    # set_dashes : solid line and space, solid line and space
    # lines[0].set_dashes([1,1,1,2,3,5])

    ax.legend()

    plt.show()


if __name__ == "__main__":
    styling2()
