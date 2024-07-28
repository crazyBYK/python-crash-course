import matplotlib.pyplot as plt
import numpy as np

m = np.linspace(0, 10, 11)
c = 3 * 10**8


def task_01():
    E = m * c**2
    plt.plot(m, E, color="red", lw=5)
    plt.title("E = mc^2")
    plt.xlabel("M in g")
    plt.ylabel("E in Jules")
    plt.xlim(0, 10)
    plt.show()


def task_01_02():
    E = m * c**2
    plt.plot(m, E, color="red", lw=5)
    plt.title("E = mc^2")
    plt.xlabel("M in g")
    plt.ylabel("E in Jules")
    plt.xlim(0, 10)
    plt.yscale("log")
    plt.grid(which="both", axis="y")
    plt.show()


labels = [
    "1 Mo",
    "3 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
]

july16_2007 = [4.75, 4.98, 5.08, 5.01, 4.89, 4.89, 4.95, 4.99, 5.05, 5.21, 5.14]
july16_2020 = [0.12, 0.11, 0.13, 0.14, 0.16, 0.17, 0.28, 0.46, 0.62, 1.09, 1.31]


def task_02():
    fig = plt.figure()
    axes = fig.add_axes([0, 0, 1, 1])

    axes.plot(labels, july16_2007, label="July16 2007")
    axes.plot(labels, july16_2020, label="July16 2020")

    plt.legend(loc="center right")
    plt.show()


def task_02_tmp():
    plt.plot(labels, july16_2007, label="July16 2007")
    plt.plot(labels, july16_2020, label="July16 2020")

    plt.legend(loc="center right")
    plt.show()


def task_02_01():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axes[0].plot(labels, july16_2007, label="July16 2007")
    axes[0].set_title("July16 2007")
    axes[1].plot(labels, july16_2020, label="July16 2020")
    axes[1].set_title("July16 2020")
    plt.show()


def task_02_bonus():
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.spines["left"].set_color("blue")
    ax1.spines["left"].set_linewidth(2)

    ax1.spines["right"].set_color("red")
    ax1.spines["right"].set_linewidth(2)

    ax1.plot(labels, july16_2007, lw=2, color="blue")
    ax1.set_ylabel("July 16 2007", fontsize=18, color="blue")
    for label in ax1.get_yticklabels():
        label.set_color("blue")

    ax2 = ax1.twinx()

    ax2.plot(labels, july16_2020, lw=2, color="red")
    ax2.set_ylabel("July 16 2020", fontsize=18, color="red")
    for label in ax2.get_yticklabels():
        label.set_color("red")

    plt.show()


if __name__ == "__main__":
    # task_01()
    # task_01_02()
    # task_02()
    # task_02_tmp()
    # task_02_01()
    task_02_bonus()
