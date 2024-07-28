def excerise():
    """what is 7 to the power of 4"""
    print(7**4)


def exercise_2():
    s = "Hi there Sam!!"
    slist = s.split()
    slist[2] = "dad!"
    print(slist)


def format_string(planet="Earth", diameter=12742):
    print(f"The diameter of {planet} is {diameter} kilometers")


def grab_hello(
    d={
        "k1": [
            1,
            2,
            3,
            {"tricky": ["oh", "man", "inception", {"target": [1, 2, 3, "hello"]}]},
        ]
    }
):
    hello = d["k1"][3]["tricky"][3]["target"][3]
    print(hello)


def grap_domain(domain="user@domain.com"):
    domain_split = domain.split("@")
    print(domain_split[-1])


def find_dog(text="Is there a dog here"):
    answer = "dog" in text
    splited = text.lower().split()
    return "dog" in splited


def count_dog(st: str) -> int:
    count = 0
    for word in st.lower().split():
        if word == "dog":
            count += 1
    return count


def final_question(speed: int, is_birthday: bool) -> str:
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed

    if speeding > 80:
        return "Big Ticket"
    elif speeding > 60:
        return "Small Ticket"
    else:
        return "No Ticket"


if __name__ == "__main__":
    grap_domain()
    format_string()
    grab_hello()

    # exercise_2()
    # print("Hello")
    #
    # print("My name is {}".format("brian"))
    # print("My name is {}, my number is {}".format("brian", "10"))
    # x = "xxx"
    # print(f"My name is {x}")
    #
    # mylist = ["a", "b", "c", "d"]
    # print(mylist[0])
    # print(mylist[-1])
    #
    # d = {"key1": "value1", "key2": "value2"}
    #
    # print(d["key2"])
    # print((1 == 1) and ("a" == "a"))
    # i = 1
    # while i < 5:
    #     print(f"i is {i}")
    #     i = i + 1
    #
    # var_answer = lambda var: var * 2
    #
    # dic = {"k1": "v1", "k2": "v2"}
    # d.keys()
    # d.items()
    #
    # mylist = [1, 2, 3]
    # last = mylist.pop()
    #
    # test = x in mylist
    # test2 = 1 in [1, 2, 3]
    #
    # excerise()
