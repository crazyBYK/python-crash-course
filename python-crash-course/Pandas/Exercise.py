import numpy as np
import pandas as pd

hotel = pd.read_csv("hotel_booking_data.csv")


def ex_01():
    # How many rows are there.
    print(hotel.info)
    print(hotel["hotel"].count())
    # print(hotel['hotel'].count())


def ex_02():
    # Is there any missing data? If so, which column has the most missing data?
    print(hotel.isnull().sum())
    max_count_none = hotel.isnull().sum().max()
    print(f"Yes, missing data, company column missing : {max_count_none} rows")

    hotel.isnull().sum()


def ex_03():
    hotel_drop_company = hotel.drop(columns="company")
    print(hotel_drop_company)
    print(hotel_drop_company.columns)

    hotels = hotel.drop("compnay", axis=1)


def ex_04():
    print(hotel["country"].value_counts()[:5])
    #     value_counts :
    #     Return a Series containing the frequency of each distinct row in the Dataframe.

    print(
        hotel.groupby("country")
        .count()
        .sort_values(by="hotel", ascending=False)["hotel"][:5]
    )

    print(hotel["country"].value_counts()[:10])


def ex_05():
    print(hotel.sort_values(by="adr", ascending=False)[["adr", "name"]].iloc[0])
    #     sort_values :
    #     Sort by the values along either axis.

    print(hotel.sort_values("adr", ascending=False)[["adr", "name"]].iloc[0])
    print(hotel["adr"].idmax())
    print(hotel.iloc[hotel["adr"].idmax()][["adr", "name"]])


def ex_06():
    # The adr is the average daily rate for a person's stay at the hotel. What is the mean adr across all the
    # hotel stays in the dataset?
    print(round(hotel["adr"].mean(), 2))
    print(round(hotel["adr"].mean(), 2))


def ex_07():
    #   What is the average (mean) number of nights for a stay across the entire data set?
    #   Feel free to round this to 2 decimal points
    hotel["total_stay_night"] = (
        hotel["stays_in_week_nights"] + hotel["stays_in_weekend_nights"]
    )
    print(round(hotel["total_stay_night"].mean(), 2))

    hotel["total_stay_night"] = (
        hotel["stays_in_week_nights"] + hotel["stays_in_weekend_nights"]
    )
    print(round(hotel["total_stay_night"].mean(), 2))


def ex_08():
    #     What is the average total cost for a stay in the dataset? Not average daily cost, but total stay cost.
    #     (You will need to calculate total cost your self by using ADR and week day and weeknight stays).
    #     Feel free to round this to 2 decimal points.
    hotel["total_stay_night"] = (
        hotel["stays_in_week_nights"] + hotel["stays_in_weekend_nights"]
    )
    hotel["total_spend"] = hotel["total_stay_night"] * hotel["adr"]
    # print(hotel['total_spend'])
    print(round(hotel["total_spend"].mean(), 2))


def ex_09():
    #     What are the names and emails of people who made exactly 5 "Special Requests"?
    print(hotel[hotel["total_of_special_requests"] == 5][["name", "email"]])


def ex_10():
    #     What percentage of hotel stays were classified as "repeat guests"?
    #     (Do not base this off the name of the person, but instead of the is_repeated_guest column)
    #     print(round(100 * sum(hotel['is_repeated_guest'] == 1) / len(hotel), 2))
    repeat_guests = round(100 * sum(hotel["is_repeated_guest"] == 1) / len(hotel), 2)
    print(repeat_guests)


def ex_11():
    #     What are the top most common last name in the dataset? Bonus: Can you figure this out
    #     line of pandas code? (For simplicity treat the a title such as MD as a last name,
    #     for example Caroline Coniey MD can be said to have the last name MD)
    print(hotel["name"].apply((lambda name: name.split()[-1])).value_counts()[:5])

    print(hotel["name"].apply((lambda name: name.split()[-1])).value_counts()[:5])


def ex_12():
    # What are the names of the people who had booked the most number children and babies for their stay?
    #     (Don't worry if they canceled, only consider number of people reported at the time of their reservation)
    print(hotel.columns)
    hotel["total_kids"] = hotel["children"] + hotel["babies"]
    print(
        hotel.sort_values("total_kids", ascending=False)[
            ["name", "adults", "total_kids", "babies", "children"]
        ][:3]
    )


def ex_13():
    #     What are the top 3 most common area code in the phone numbers? (Area code is first 3 digits)
    #     print(hotel['phone-number'])
    print(hotel["phone-number"].apply(lambda phone: phone[:3]).value_counts()[:3])
    # print(hotel['phone-number'].apply(lambda num: num[:3]).value_counts()[:3])
    print(hotel["phone-number"].apply(lambda phone: phone[:3]).value_counts()[:3])


def ex_14():
    #  How many arrivals took place between the 1st and the 15th of the month(inclusive of 1 and 15)?
    #  Bonus: Can you do this in one line of pandas code?
    #     print(hotel['arrival_date_day_of_month'].apply(lambda date : date in range(1, 16)).sum())
    print(
        hotel["arrival_date_day_of_month"]
        .apply(lambda data: data in range(1, 16))
        .sum()
    )


def ex_15():
    #     HARD BONUS TASK: Create a table for counts for each day of the week that people arrived.
    #     (E.g. 5000 arrivals were on a Monday, 3000 were on a Tuesday. etc...)
    #     print(hotel['arrival_date_day_of_month'])
    print(hotel.columns)
    print(hotel["arrival_date_week_number"])

    # using vectorize to convert year, month, date into date format
    hotel["date"] = np.vectorize(convert_date)(
        hotel["arrival_date_year"],
        hotel["arrival_date_month"],
        hotel["arrival_date_day_of_month"],
    )
    print(hotel["date"])

    # casting string to pandas datetime format
    hotel["date"] = pd.to_datetime(hotel["date"])

    # counting value by day name.
    print(hotel["date"].dt.day_name().value_counts())


def convert_date(year: int, month: int, day: int):
    return f"{day}-{month}-{year}"


if __name__ == "__main__":
    print("Start Exercise..!!")
    # ex_01()
    # ex_02()
    # ex_03()
    # ex_04()
    # ex_05()
    # ex_06()
    # ex_07()
    # ex_08()
    # ex_09()
    # ex_10()
    # ex_11()
    # ex_12()
    # ex_13()
    # ex_14()
    ex_15()
