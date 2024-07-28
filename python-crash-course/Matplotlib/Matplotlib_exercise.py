import matplotlib.pyplot as plt
import numpy as np


def part_01_ex_01():
    #     Use your knowledge of Numpy to create two arrays: E and m, where m is simply 11 evenly spaced values
    #     representing 0 grams to 10 grams. E should be the equivalent energy for the mass.
    #     You will need to figure out what to provide for c for the units m/s, a quick google search will
    #     easily give you the answer (we'll use the close approximation in our solutions).
    m = np.linspace(0, 10, 11)
    print(m)
    # Mass–energy equivalence
    c = 2.99792458e8
    e = m * c**2
    print(e)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(m, e)

    ax.set_title("E=mc^2")
    ax.set_ylabel("Energy in Joules")
    ax.set_xlabel("Mass in Grams")

    # plot chart on a logarthimic scale on the y axis?
    plt.yscale("log")

    plt.title("E=mc^2")
    plt.xlabel("Mass in Grams")
    plt.ylabel("Energy in Joules")
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


def part_02_ex_01():
    # We've obtained some yelid curve data for you form the US Treasury Dept..
    # The data shows the interest paid for a US Treasury bond for a certain
    # contract length. The labels list shows teh corresponding contract length per index position

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(labels, july16_2007, color="blue", label="July16 2007")
    ax.plot(labels, july16_2020, color="orange", label="July16 2020")
    plt.legend(loc="center right")

    plt.show()


def part_02_ex_02():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(labels, july16_2007, color="blue", label="July16 2007")
    ax.plot(labels, july16_2020, color="orange", label="July16 2022")

    # Figure out how to plot both curves on the same Figure. Add a legend to
    # show which cuvre corresponds to a certain year..
    plt.legend(loc="center right")
    plt.show()


#     The legend in the plot above looks a littole strange in the middle of the curves.
#     while it is not blocking anything, it would be nicer if it were outside the plot.
#     plot. Figure out how to move the legend ourside the amin Figure plot.
def part_02_ex_03():
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(labels, july16_2007, label="July16 2007", color="blue")
    ax.plot(labels, july16_2020, label="july16 2022", color="orange")

    plt.legend(loc=(1.1, 0.5))
    plt.show()


# while the plot above clearly shows how rates fell from 2007 to 20202, putting these on the same plot makes
# it difficult to descern the rate differences within the same year. Use .subplot() to
# create the plot figure below. which show s each years yield curve.


def part_02_ex_04():
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    fig, aces = plt.subplots(nrows=2, figsize=(), dpi=300)


if __name__ == "__main__":
    # part_01_ex_01()
    # part_02_ex_01()
    # part_02_ex_02()
    # part_02_ex_03()
    part_02_ex_04()
