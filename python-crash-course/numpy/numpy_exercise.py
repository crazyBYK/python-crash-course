import numpy as np


def ex01():
    mylist = [1, 2, 3]
    print(type(mylist))

    myarr = np.array(mylist)
    print(type(myarr))

    my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    np_my_matrix = np.array(my_matrix)
    print(np_my_matrix)

    print(np.arange(0, 10))

    print(np.zeros(5))
    print(np.zeros((2, 5)))
    print(np.ones((4, 4)))
    print(np.linspace(0, 10, 10))
    print(np.eye(5))

    print(np.random.rand(5, 2))
    print(np.random.randn(10, 2))
    print(np.random.randint(1, 46, (5, 5, 5)))
    print(np.random.seed(42))

    arr = np.arange(0, 25)
    print(arr.reshape(5, 5))
    ranarr = np.random.randint(0, 101, 10)
    print(ranarr.max())
    print(ranarr.min())
    print(ranarr.argmax())
    print(ranarr.dtype)

    print(arr.shape)
    arr = arr.reshape(5, 5)


def ex_indexing_selection():
    arr = np.arange(0, 11)
    print(arr[8])
    print(arr[1:5])
    print(arr[0:5])
    print(arr[:5])
    print(arr[5:])

    arr[0:5] = 100
    print(arr)

    slice_of_arr = arr[0:5]
    slice_of_arr[:] = 99
    """slice of arr pointed to original arr. so if you broadcast slice of arr, original arr will broadcast too"""
    print(slice_of_arr)
    print(arr)

    slice_of_arr_copy = slice_of_arr.copy()
    slice_of_arr_copy[:] = 1

    print(slice_of_arr_copy)
    print(slice_of_arr)
    print(arr)

    arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])

    print(arr_2d[1][1])
    print(arr_2d[1, 1])
    print(arr_2d[1, :2])
    print(arr_2d[:2, 1:])

    arr = np.arange(1, 11)
    print(arr)
    print(arr > 4)
    bool_arr = arr > 4
    print(bool_arr)
    print(arr[bool_arr])
    print(arr[arr > 4])


def numpy_operator():
    arr = np.arange(0, 10)
    print(arr)

    print(arr + 2)
    print(arr + arr)
    print(arr - arr)
    # print(arr / arr)
    print(np.sqrt(arr))
    print(np.sin(arr))
    # print(np.log(arr))

    arr_2d = np.arange(0, 25).reshape(5, 5)
    print(arr_2d)

    # sum across the rows
    print(arr_2d.sum(axis=0))

    # sum across the columns
    print(arr_2d.sum(axis=1))


if __name__ == "__main__":
    # ex_indexing_selection()
    numpy_operator()
    print(np.array(10))

    print(np.arange(10, 50))
    print(np.linspace(10, 50, 21))
    print(np.arange(0, 9).reshape(3, 3))
    print(np.random.randn(1))

    print(np.ones(10) * 5)

    arr = np.arange(1, 101) / 100
    print(arr.reshape(10, 10))

    mat = np.arange(1, 26).reshape(5, 5)
    print(mat)
    print(mat[2:, 1:])
    print(mat[:3, 1:2])
    print(mat[4])
