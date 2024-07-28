import pandas as pd


def read_excel_input():
    df = pd.read_excel("my_excel_file.xlsx", sheet_name="First_Sheet")
    print(df)
    wb = pd.ExcelFile("my_excel_file.xlsx")
    print(wb.sheet_names)

    excel_sheet_dict = pd.read_excel("my_excel_file.xlsx", sheet_name=None)
    print(type(excel_sheet_dict))
    print(excel_sheet_dict.keys())
    print(excel_sheet_dict["First_Sheet"])

    our_df = excel_sheet_dict["First_Sheet"]
    print(our_df)

    our_df.to_excel("example.xlsx", sheet_name="First_Sheet", index=False)


if __name__ == "__main__":
    read_excel_input()
